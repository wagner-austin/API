package rwbot.agent;

/**
 * Builds the replacement {@code Code} attribute bodies the patcher splices in.
 *
 * <p>Split from {@link ClassFilePatcher} when the retarget edit pushed that
 * file past the module ceiling: this half is "what a legal replacement body
 * looks like" -- pure functions of a method descriptor -- and the patcher
 * half is "how the class file is walked and edited". The JVM verifier
 * remains the oracle for both, through {@link SelfTest}.
 */
final class CodeBodies {

    private static final int ACC_STATIC = 0x0008;

    private CodeBodies() {
    }

    /**
     * A complete Code attribute body (attribute_length included) holding the
     * smallest legal body that satisfies {@code descriptor}'s return type.
     */
    static byte[] noOp(String descriptor, int accessFlags) {
        byte[] code = returnSequence(descriptor);
        int maxStack = maxStackFor(descriptor);
        int maxLocals = argumentSlots(descriptor) + ((accessFlags & ACC_STATIC) != 0 ? 0 : 1);

        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        int attributeLength = 2 + 2 + 4 + code.length + 2 + 2;
        writeU4(out, attributeLength);
        writeU2(out, maxStack);
        writeU2(out, maxLocals);
        writeU4(out, code.length);
        out.write(code, 0, code.length);
        writeU2(out, 0); // exception_table_length
        writeU2(out, 0); // attributes_count
        return out.toByteArray();
    }

    /**
     * A complete Code attribute whose body is {@code this.<delegate>();
     * return}. Straight-line, so no StackMapTable is needed at any class file
     * version; one reference on the stack, so max_stack is one.
     */
    static byte[] delegate(String descriptor, int accessFlags, int methodRef) {
        byte[] code = {
            (byte) 0x2a, // aload_0
            (byte) 0xb6, // invokevirtual
            (byte) ((methodRef >>> 8) & 0xff),
            (byte) (methodRef & 0xff),
            (byte) 0xb1, // return
        };
        int maxLocals = argumentSlots(descriptor) + ((accessFlags & ACC_STATIC) != 0 ? 0 : 1);

        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        int attributeLength = 2 + 2 + 4 + code.length + 2 + 2;
        writeU4(out, attributeLength);
        writeU2(out, 1); // max_stack
        writeU2(out, maxLocals);
        writeU4(out, code.length);
        out.write(code, 0, code.length);
        writeU2(out, 0); // exception_table_length
        writeU2(out, 0); // attributes_count
        return out.toByteArray();
    }

    /** The minimal instruction sequence returning a default value of the descriptor's return type. */
    private static byte[] returnSequence(String descriptor) {
        char returnType = descriptor.charAt(descriptor.indexOf(')') + 1);
        switch (returnType) {
            case 'V':
                return new byte[] {(byte) 0xb1}; // return
            case 'Z':
            case 'B':
            case 'C':
            case 'S':
            case 'I':
                return new byte[] {(byte) 0x03, (byte) 0xac}; // iconst_0; ireturn
            case 'J':
                return new byte[] {(byte) 0x09, (byte) 0xad}; // lconst_0; lreturn
            case 'F':
                return new byte[] {(byte) 0x0b, (byte) 0xae}; // fconst_0; freturn
            case 'D':
                return new byte[] {(byte) 0x0e, (byte) 0xaf}; // dconst_0; dreturn
            case 'L':
            case '[':
                return new byte[] {(byte) 0x01, (byte) 0xb0}; // aconst_null; areturn
            default:
                throw new ClassFormatError(
                        "unsupported return type '" + returnType + "' in " + descriptor);
        }
    }

    private static int maxStackFor(String descriptor) {
        char returnType = descriptor.charAt(descriptor.indexOf(')') + 1);
        switch (returnType) {
            case 'V':
                return 0;
            case 'J':
            case 'D':
                return 2;
            default:
                return 1;
        }
    }

    /** Counts argument slots in a method descriptor; long and double take two. */
    private static int argumentSlots(String descriptor) {
        int slots = 0;
        int i = descriptor.indexOf('(') + 1;
        while (descriptor.charAt(i) != ')') {
            char c = descriptor.charAt(i);
            if (c == '[') {
                i++;
                continue;
            }
            if (c == 'L') {
                i = descriptor.indexOf(';', i) + 1;
                slots++;
                continue;
            }
            slots += (c == 'J' || c == 'D') ? 2 : 1;
            i++;
        }
        return slots;
    }

    static void writeU2(java.io.ByteArrayOutputStream out, int value) {
        out.write((value >>> 8) & 0xff);
        out.write(value & 0xff);
    }

    static void writeU4(java.io.ByteArrayOutputStream out, int value) {
        out.write((value >>> 24) & 0xff);
        out.write((value >>> 16) & 0xff);
        out.write((value >>> 8) & 0xff);
        out.write(value & 0xff);
    }
}
