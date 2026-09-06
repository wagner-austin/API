package rwbot.agent;

/**
 * Instruction-level reads over a raw class-file buffer.
 *
 * <p>Split from {@link ClassFilePatcher} at the module ceiling, along the
 * same seam as {@link CodeBodies}: the patcher owns how a file is edited,
 * this owns how its bytecode is measured -- instruction lengths for the
 * walk, and the {@code LineNumberTable} resolution the line-scoped retarget
 * filters through. Pure functions of the buffer; nothing here writes.
 */
final class Bytecode {

    private Bytecode() {
    }

    /**
     * The byte length of one instruction, including its opcode.
     *
     * <p>Complete over the instruction set the pinned jar's methods use, with
     * the three variable-length shapes ({@code wide}, {@code tableswitch},
     * {@code lookupswitch}) computed rather than refused: a walker that bails
     * on a switch would quietly narrow which methods can ever be retargeted.
     *
     * @param buf The class-file bytes.
     * @param opcode The instruction's first byte.
     * @param codeStart Buffer offset of the code array's first byte, from
     *     which the switch shapes compute their 4-byte alignment padding.
     * @param at The instruction's offset within the code array.
     * @throws ClassFormatError on an opcode outside the JVMS table -- a
     *     mis-walk would rewrite arbitrary bytes, so unknown means stop.
     */
    static int instructionLength(byte[] buf, int opcode, int codeStart, int at) {
        if (opcode <= 0x0f || (opcode >= 0x1a && opcode <= 0x35)
                || (opcode >= 0x3b && opcode <= 0x83 && opcode != 0x84)
                || (opcode >= 0x85 && opcode <= 0x98)
                || (opcode >= 0xac && opcode <= 0xb1)
                || opcode == 0x5f || opcode == 0xbe || opcode == 0xbf
                || opcode == 0xc2 || opcode == 0xc3) {
            return 1;
        }
        switch (opcode) {
            case 0x10: // bipush
            case 0x12: // ldc
            case 0x15: case 0x16: case 0x17: case 0x18: case 0x19: // loads
            case 0x36: case 0x37: case 0x38: case 0x39: case 0x3a: // stores
            case 0xbc: // newarray
                return 2;
            case 0x11: // sipush
            case 0x13: case 0x14: // ldc_w, ldc2_w
            case 0x84: // iinc
            case 0xb2: case 0xb3: case 0xb4: case 0xb5: // get/putstatic, get/putfield
            case 0xb6: case 0xb7: case 0xb8: // invokevirtual/special/static
            case 0xbb: // new
            case 0xbd: // anewarray
            case 0xc0: case 0xc1: // checkcast, instanceof
            case 0xc6: case 0xc7: // ifnull, ifnonnull
                return 3;
            case 0xc5: // multianewarray
                return 4;
            case 0xb9: // invokeinterface
            case 0xba: // invokedynamic
            case 0xc8: case 0xc9: // goto_w, jsr_w
                return 5;
            case 0xc4: // wide
                return (buf[codeStart + at + 1] & 0xff) == 0x84 ? 6 : 4;
            case 0xaa: { // tableswitch
                int aligned = (at + 4 + 3) & ~3;
                int low = readInt(buf, codeStart + aligned + 4);
                int high = readInt(buf, codeStart + aligned + 8);
                return (aligned - at) + 12 + (high - low + 1) * 4;
            }
            case 0xab: { // lookupswitch
                int aligned = (at + 4 + 3) & ~3;
                int pairs = readInt(buf, codeStart + aligned + 4);
                return (aligned - at) + 8 + pairs * 8;
            }
            default:
                if (opcode >= 0x99 && opcode <= 0xa8) { // ifs, goto, jsr
                    return 3;
                }
                if (opcode == 0xa9) { // ret
                    return 2;
                }
                throw new ClassFormatError(
                        "unmodeled opcode 0x" + Integer.toHexString(opcode)
                                + " at code offset " + at);
        }
    }

    /** A big-endian s4 read at an absolute buffer offset, for the switch shapes. */
    static int readInt(byte[] buf, int offset) {
        return ((buf[offset] & 0xff) << 24)
                | ((buf[offset + 1] & 0xff) << 16)
                | ((buf[offset + 2] & 0xff) << 8)
                | (buf[offset + 3] & 0xff);
    }

    /** A big-endian u2 read at an absolute buffer offset. */
    static int readShort(byte[] buf, int offset) {
        return ((buf[offset] & 0xff) << 8) | (buf[offset + 1] & 0xff);
    }

    /**
     * Reads every {@code LineNumberTable} inside one Code attribute, merged.
     *
     * <p>Positional reads against absolute offsets, so the caller's own file
     * cursor is untouched.
     *
     * @param buf The class-file bytes.
     * @param pool The constant pool's UTF-8 entries by index.
     * @param codeStart Buffer offset of the code array's first byte.
     * @param codeLength The code array's length.
     * @return Two parallel arrays: entry start_pcs and their line numbers, in
     *     file order.
     */
    static int[][] readLineTable(byte[] buf, String[] pool, int codeStart, int codeLength) {
        int cursor = codeStart + codeLength;
        int exceptionCount = readShort(buf, cursor);
        cursor += 2 + exceptionCount * 8;
        int attributeCount = readShort(buf, cursor);
        cursor += 2;
        java.util.List<int[]> entries = new java.util.ArrayList<int[]>();
        for (int i = 0; i < attributeCount; i++) {
            String attributeName = pool[readShort(buf, cursor)];
            int attributeLength = readInt(buf, cursor + 2);
            int body = cursor + 6;
            if ("LineNumberTable".equals(attributeName)) {
                int count = readShort(buf, body);
                for (int e = 0; e < count; e++) {
                    int entry = body + 2 + e * 4;
                    entries.add(
                            new int[] {readShort(buf, entry), readShort(buf, entry + 2)});
                }
            }
            cursor = body + attributeLength;
        }
        int[] starts = new int[entries.size()];
        int[] numbers = new int[entries.size()];
        for (int i = 0; i < entries.size(); i++) {
            starts[i] = entries.get(i)[0];
            numbers[i] = entries.get(i)[1];
        }
        return new int[][] {starts, numbers};
    }

    /**
     * The source line owning the instruction at {@code at}: the table entry
     * with the largest start_pc at or below it (JVMS 4.7.12 maps ranges this
     * way), or -1 with no table entry at or below.
     */
    static int lineAt(int[] lineStarts, int[] lineNumbers, int at) {
        int line = -1;
        int best = -1;
        for (int i = 0; i < lineStarts.length; i++) {
            if (lineStarts[i] <= at && lineStarts[i] >= best) {
                best = lineStarts[i];
                line = lineNumbers[i];
            }
        }
        return line;
    }
}
