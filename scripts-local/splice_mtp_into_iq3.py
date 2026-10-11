#!/usr/bin/env python3
"""
Splice native MTP (layer 48) tensors from Q2 model into IQ3_XXS model.
Produces a new standalone shard 1 with 49 layers (48 base + 1 MTP)
and hardlinks/adapts shard 2.
"""

import os
import sys
import time
import struct
from pathlib import Path

# Base paths
IQ3_DIR = Path("/Users/localadmin/model_dir/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/IQ3_XXS/IQ3_XXS")
IQ3_SHARD1 = IQ3_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
IQ3_SHARD2 = IQ3_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf"

Q2_FILE = Path("/Users/localadmin/github-sources/ds4/gguf/Qwen3.8-Flash-Next-Q2.gguf")

OUT_DIR = Path("/Users/localadmin/model_dir/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/IQ3_XXS_MTP")
OUT_SHARD1 = OUT_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS_MTP-00001-of-00002.gguf"
OUT_SHARD2 = OUT_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS_MTP-00002-of-00002.gguf"

def main():
    print(f"=== Starting GGUF MTP Splicer ===")
    print(f"Source IQ3 Shard 1: {IQ3_SHARD1}")
    print(f"Source IQ3 Shard 2: {IQ3_SHARD2}")
    print(f"Source Q2 File:     {Q2_FILE}")
    print(f"Output Directory:   {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Inspect Q2 MTP tensors metadata and offsets
    print("\n[1/6] Scanning Q2 blk.48 tensors...")
    sys.path.insert(0, "/Users/localadmin/github-sources/llama.cpp/gguf-py")
    import gguf

    reader_q2 = gguf.GGUFReader(str(Q2_FILE))
    mtp_tensors = [t for t in reader_q2.tensors if "blk.48." in t.name]
    assert len(mtp_tensors) == 32, f"Expected 32 MTP tensors, got {len(mtp_tensors)}"

    t_first = reader_q2.tensors[1223]
    t_last = reader_q2.tensors[1254]
    t_next = reader_q2.tensors[1255]

    q2_ti_offset = t_first.field.offset
    q2_ti_len = t_next.field.offset - t_first.field.offset
    assert q2_ti_len == 2065, f"Unexpected Q2 TI len: {q2_ti_len}"

    q2_payload_start = t_first.data_offset
    q2_payload_len = (t_last.data_offset + t_last.n_bytes) - t_first.data_offset
    print(f"  Found 32 MTP tensors. Total payload size: {q2_payload_len:,} bytes ({q2_payload_len/(1024*1024):.2f} MB)")

    # Read raw Q2 TI bytes
    with open(Q2_FILE, "rb") as fq2:
        fq2.seek(q2_ti_offset)
        raw_q2_ti = bytearray(fq2.read(q2_ti_len))

    # 2. Inspect IQ3 Shard 1 header and geometry
    print("\n[2/6] Reading IQ3 Shard 1 header...")
    with open(IQ3_SHARD1, "rb") as fiq3:
        # Read entire header up to data_offset (11,024,256 bytes)
        raw_header = bytearray(fiq3.read(11024256))

    # Verify header magic and counts
    magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", raw_header[:24])
    assert magic == b"GGUF", "Invalid GGUF magic"
    assert version == 3, f"Unexpected GGUF version: {version}"
    assert tensor_count == 1223, f"Unexpected tensor_count: {tensor_count}"
    assert kv_count == 67, f"Unexpected kv_count: {kv_count}"

    # Verify key positions in header
    pos_block_count = raw_header.find(b"qwen4exp.block_count\x04\x00\x00\x00\x30\x00\x00\x00")
    assert pos_block_count != -1, "Could not find qwen4exp.block_count in IQ3 header"
    print(f"  Found qwen4exp.block_count at offset {pos_block_count}")

    pos_split_tensors = raw_header.find(b"split.tensors.count\x05\x00\x00\x00\xc8\x04\x00\x00")
    assert pos_split_tensors != -1, "Could not find split.tensors.count in IQ3 header"
    print(f"  Found split.tensors.count at offset {pos_split_tensors}")

    pos_kv_end = 10946629
    assert raw_header[pos_kv_end : pos_kv_end + 8] == struct.pack("<Q", 13), "Unexpected start of tensor infos"

    # 3. Construct new GGUF header for Spliced Shard 1
    print("\n[3/6] Constructing modified GGUF header and tensor info table...")
    new_tensor_count = 1255  # 1223 + 32
    new_kv_count = 68       # 67 + 1 (qwen4exp.nextn_predict_layers)

    # Slice header components:
    # Header Prefix (0 .. 8): magic + version
    hdr_prefix = raw_header[:8]
    # Encoded counts (8 .. 24)
    hdr_counts = struct.pack("<QQ", new_tensor_count, new_kv_count)
    # KV section before split.tensors.count
    kv_part1 = raw_header[24 : pos_block_count]
    # Update block_count from 48 to 49
    kv_block_count = b"qwen4exp.block_count\x04\x00\x00\x00\x31\x00\x00\x00"

    # Find and update qwen4exp.attention.compress_ratios from 48 elements to 49 elements
    pos_compress = raw_header.find(b"qwen4exp.attention.compress_ratios")
    assert pos_compress != -1, "Could not find qwen4exp.attention.compress_ratios in IQ3 header"
    field_compress_start = pos_compress - 8
    p = pos_compress + 34 + 16
    field_compress_end = p + 48 * 4  # 48 UINT32s

    kv_part2_a = raw_header[pos_block_count + len(kv_block_count) : field_compress_start]
    
    # 49 elements for compress_ratios (append 4 for MTP layer 48)
    compress_name = b"qwen4exp.attention.compress_ratios"
    compress_arr = [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 4]
    kv_compress = struct.pack("<Q", len(compress_name)) + compress_name + struct.pack("<IIQ", 9, 5, len(compress_arr)) + b"".join(struct.pack("<I", x) for x in compress_arr)

    kv_part2_b = raw_header[field_compress_end : pos_split_tensors]

    # Update split.tensors.count from 1224 to 1256 (1255 in shard 1 + 1 in shard 2)
    kv_split_tensors = b"split.tensors.count\x05\x00\x00\x00\xe8\x04\x00\x00"
    kv_part3 = raw_header[pos_split_tensors + len(kv_split_tensors) : pos_kv_end]
    # Add new KV pair: qwen4exp.nextn_predict_layers = 1 (UINT32)
    kv_nextn = b"\x1d\x00\x00\x00\x00\x00\x00\x00qwen4exp.nextn_predict_layers\x04\x00\x00\x00\x01\x00\x00\x00"

    # Tensor Info section:
    # 1223 existing IQ3 tensor info entries (10,946,629 .. 11,024,251)
    ti_iq3 = raw_header[pos_kv_end : 11024251]
    assert len(ti_iq3) == 77622, f"Unexpected ti_iq3 length: {len(ti_iq3)}"

    # Patch Q2 MTP Tensor Info relative offsets:
    # In IQ3, the 1223 tensors occupy rel_offsets 0 .. 47,331,120,640 (aligned to 32)
    base_rel_offset_mtp = 47331120640
    orig_q2_first_rel_offset = 43287498240
    delta_rel_offset = base_rel_offset_mtp - orig_q2_first_rel_offset

    pos = 0
    for i in range(32):
        name_len = struct.unpack("<Q", raw_q2_ti[pos:pos+8])[0]
        pos += 8 + name_len
        n_dims = struct.unpack("<I", raw_q2_ti[pos:pos+4])[0]
        pos += 4 + n_dims * 8 + 4
        rel_offset = struct.unpack("<Q", raw_q2_ti[pos:pos+8])[0]
        new_rel_offset = rel_offset + delta_rel_offset
        raw_q2_ti[pos:pos+8] = struct.pack("<Q", new_rel_offset)
        pos += 8

    # Assemble new header up to end of tensor infos
    new_hdr_unaligned = bytearray()
    new_hdr_unaligned.extend(hdr_prefix)
    new_hdr_unaligned.extend(hdr_counts)
    new_hdr_unaligned.extend(kv_part1)
    new_hdr_unaligned.extend(kv_block_count)
    new_hdr_unaligned.extend(kv_part2_a)
    new_hdr_unaligned.extend(kv_compress)
    new_hdr_unaligned.extend(kv_part2_b)
    new_hdr_unaligned.extend(kv_split_tensors)
    new_hdr_unaligned.extend(kv_part3)
    new_hdr_unaligned.extend(kv_nextn)
    new_hdr_unaligned.extend(ti_iq3)
    new_hdr_unaligned.extend(raw_q2_ti)

    # Pad header to 32-byte boundary
    pad_len = (32 - (len(new_hdr_unaligned) % 32)) % 32
    new_hdr_unaligned.extend(b"\x00" * pad_len)
    new_data_offset = len(new_hdr_unaligned)
    print(f"  New Header constructed: {new_data_offset:,} bytes (alignment: 32)")

    # Check shift in data_offset
    old_data_offset = 11024256
    shift = new_data_offset - old_data_offset
    print(f"  Header shift: {shift} bytes ({old_data_offset} -> {new_data_offset})")

    if shift != 0:
        # If header length changed, all tensor relative offsets in ti_iq3 and raw_q2_ti
        # would need adjustment ONLY IF relative offset was from file start.
        # But in GGUF v3, tensor data offset is relative to data_offset (the end of header padding)!
        # So rel_offsets are completely invariant to header padding shifts!
        print(f"  NOTE: GGUF relative offsets are measured relative to data_offset, so existing rel_offsets remain valid.")

    # 4. Stream and write new Shard 1
    print("\n[4/6] Streaming Shard 1 data...")
    t0 = time.time()
    chunk_size = 32 * 1024 * 1024  # 32 MB chunks

    with open(OUT_SHARD1, "wb") as fout:
        # Write new header
        fout.write(new_hdr_unaligned)

        # Stream base payload from IQ3 Shard 1 (from old_data_offset to EOF)
        print("  Streaming base IQ3 layers (0-47)...")
        with open(IQ3_SHARD1, "rb") as fin_iq3:
            fin_iq3.seek(old_data_offset)
            total_iq3_bytes = 0
            while True:
                buf = fin_iq3.read(chunk_size)
                if not buf:
                    break
                fout.write(buf)
                total_iq3_bytes += len(buf)
                if total_iq3_bytes % (2 * 1024 * 1024 * 1024) < chunk_size:
                    print(f"    Copied {total_iq3_bytes / (1024*1024*1024):.1f} GB base weights...")

        print(f"  Finished base weights ({total_iq3_bytes:,} bytes).")

        # Stream MTP payload from Q2
        print("  Streaming layer 48 MTP weights from Q2...")
        with open(Q2_FILE, "rb") as fin_q2:
            fin_q2.seek(q2_payload_start)
            bytes_left = q2_payload_len
            while bytes_left > 0:
                to_read = min(bytes_left, chunk_size)
                buf = fin_q2.read(to_read)
                if not buf:
                    break
                fout.write(buf)
                bytes_left -= len(buf)

        print(f"  Finished MTP weights ({q2_payload_len:,} bytes).")

    elapsed = time.time() - t0
    final_shard1_size = OUT_SHARD1.stat().st_size
    print(f"  Shard 1 written successfully in {elapsed:.1f}s. Size: {final_shard1_size:,} bytes ({final_shard1_size/(1024*1024*1024):.2f} GB)")

    # 5. Create Shard 2 (update split.tensors.count in header)
    print("\n[5/6] Creating adapted Shard 2...")
    with open(IQ3_SHARD2, "rb") as fs2:
        s2_hdr = bytearray(fs2.read(1024 * 1024))
        # Search for split.tensors.count
        pos_s2_split = s2_hdr.find(b"split.tensors.count\x05\x00\x00\x00\xc8\x04\x00\x00")
        assert pos_s2_split != -1, "Could not find split.tensors.count in Shard 2"
        # In-place patch from 1224 (0x04c8) to 1256 (0x04e8)
        s2_hdr[pos_s2_split + 23 : pos_s2_split + 27] = struct.pack("<I", 1256)

        # Check shard 2 header size vs payload
        fs2.seek(0)
        reader_s2 = gguf.GGUFReader(str(IQ3_SHARD2))
        s2_data_offset = reader_s2.data_offset

    # Write modified Shard 2
    # Note: Shard 2 header is small (~1KB), payload is 26.8 GB.
    # To avoid copying 26.8 GB of unchanged PLE weights, check if we can hardlink or if we copy.
    # If we copy, it takes ~20 seconds on NVMe. Let's write Shard 2 header and stream payload.
    print(f"  Writing adapted Shard 2 (header: {s2_data_offset} bytes)...")
    t0_s2 = time.time()
    with open(OUT_SHARD2, "wb") as fout_s2:
        with open(IQ3_SHARD2, "rb") as fin_s2:
            # Read exact original header
            orig_s2_hdr = bytearray(fin_s2.read(s2_data_offset))
            # Patch split.tensors.count in orig header
            p = orig_s2_hdr.find(b"split.tensors.count\x05\x00\x00\x00\xc8\x04\x00\x00")
            assert p != -1
            orig_s2_hdr[p + 23 : p + 27] = struct.pack("<I", 1256)
            fout_s2.write(orig_s2_hdr)

            # Stream payload
            total_s2 = 0
            while True:
                buf = fin_s2.read(chunk_size)
                if not buf:
                    break
                fout_s2.write(buf)
                total_s2 += len(buf)
                if total_s2 % (4 * 1024 * 1024 * 1024) < chunk_size:
                    print(f"    Copied {total_s2 / (1024*1024*1024):.1f} GB Shard 2 PLE weights...")

    elapsed_s2 = time.time() - t0_s2
    print(f"  Shard 2 completed in {elapsed_s2:.1f}s.")

    # 6. Verify newly created shards with GGUFReader
    print("\n[6/6] Validating spliced model files...")
    reader_out1 = gguf.GGUFReader(str(OUT_SHARD1))
    print(f"  Shard 1 tensor count: {len(reader_out1.tensors)} (expected 1255)")
    print(f"  Shard 1 block_count:  {reader_out1.fields['qwen4exp.block_count'].contents()}")
    print(f"  Shard 1 nextn_layers: {reader_out1.fields['qwen4exp.nextn_predict_layers'].contents()}")
    print(f"  Shard 1 split tensors:{reader_out1.fields['split.tensors.count'].contents()}")

    reader_out2 = gguf.GGUFReader(str(OUT_SHARD2))
    print(f"  Shard 2 tensor count: {len(reader_out2.tensors)} (expected 1)")
    print(f"  Shard 2 split tensors:{reader_out2.fields['split.tensors.count'].contents()}")

    print("\n=== Splicing and Validation COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
