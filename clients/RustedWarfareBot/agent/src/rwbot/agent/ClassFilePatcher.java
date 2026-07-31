package rwbot.agent;

/**
 * Replaces named method bodies in a class file with minimal no-op bodies.
 *
 * <p>Dependency-free by construction. The agent loads into the game's own
 * classloader beside obfuscated 1.15 classes, so every additional runtime
 * dependency is a conflict surface; a bytecode library is not worth it for an
 * edit this narrow.
 *
 * <p>The edit is local to a single {@code Code} attribute. A no-op body
 * references no constant-pool entries, so the pool is parsed only to find
 * where it ends and is copied through byte-for-byte. Nothing outside the
 * replaced attribute changes, and no length field outside it needs recomputing:
 * {@code method_info} carries no total length, and the class file itself has no
 * trailing size. That is what keeps this small enough to hand-roll.
 *
 * <p>The JVM's own bytecode verifier is the oracle that this is correct --
 * see {@link SelfTest}, which defines every patched class rather than merely
 * inspecting it.
 */
final class ClassFilePatcher {

    private static final int CONSTANT_UTF8 = 1;
    private static final int CONSTANT_INTEGER = 3;
    private static final int CONSTANT_FLOAT = 4;
    private static final int CONSTANT_LONG = 5;
    private static final int CONSTANT_DOUBLE = 6;
    private static final int CONSTANT_CLASS = 7;
    private static final int CONSTANT_STRING = 8;
    private static final int CONSTANT_FIELDREF = 9;
    private static final int CONSTANT_METHODREF = 10;
    private static final int CONSTANT_INTERFACE_METHODREF = 11;
    private static final int CONSTANT_NAME_AND_TYPE = 12;
    private static final int CONSTANT_METHOD_HANDLE = 15;
    private static final int CONSTANT_METHOD_TYPE = 16;
    private static final int CONSTANT_DYNAMIC = 17;
    private static final int CONSTANT_INVOKE_DYNAMIC = 18;
    private static final int CONSTANT_MODULE = 19;
    private static final int CONSTANT_PACKAGE = 20;

    private static final int ACC_STATIC = 0x0008;

    private final byte[] buf;
    private int pos;

    // The structured pool view, populated by readHeaderAndConstantPool. Only
    // Class, Methodref and NameAndType operands are meaningful; everything
    // else keeps zeros. Needed only to find an existing self-call for
    // delegation -- the no-op path never reads them.
    private int[] tags;
    private int[] operandA;
    private int[] operandB;

    // Constant-pool index of the Methodref a delegating body invokes, or -1
    // when patching to plain no-ops.
    private int delegateRef = -1;

    private ClassFilePatcher(byte[] buf) {
        this.buf = buf;
        this.pos = 0;
    }

    /**
     * Returns a copy of {@code classFile} with the body of every method named
     * in {@code targets} replaced by a no-op, or {@code null} if no target
     * matched. A null return means "leave this class alone" -- the contract a
     * {@link java.lang.instrument.ClassFileTransformer} already expects.
     *
     * @throws ClassFormatError if the class file cannot be parsed. Failures
     *     propagate rather than being softened into a silent skip: a parse
     *     failure means the format assumption is wrong, and quietly declining
     *     to patch would surface later as the original NullPointerException
     *     with no indication why the agent did nothing.
     */
    static byte[] noOpMethods(byte[] classFile, java.util.Set<String> targets) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        return patcher.patch(targets, null, null);
    }

    /**
     * Returns a copy of {@code classFile} with the body of the method named by
     * {@code target} (name plus descriptor, e.g. {@code "a()V"}) replaced by
     * {@code this.<delegate>(); return;}, or {@code null} if the target did
     * not match.
     *
     * <p>No constant-pool entry is added: the class must already invoke the
     * delegate on itself somewhere, so the Methodref this body needs is found
     * rather than forged. That keeps the edit as local as the no-op -- the
     * pool is copied through byte-for-byte, same as ever.
     *
     * <p>Both the target and the delegate must be instance methods returning
     * void: the emitted body is {@code aload_0; invokevirtual; return}, which
     * is only verifiable under exactly those shapes.
     *
     * @throws ClassFormatError if the class cannot be parsed, the delegate
     *     Methodref is absent from the pool, or a matched target is static or
     *     non-void. All of these mean the pinned engine build moved under the
     *     patch, and a loud failure at class load beats a silent skip.
     */
    static byte[] delegateToSelf(
            byte[] classFile, String target, String delegateName, String delegateDescriptor) {
        if (!delegateDescriptor.endsWith(")V") || !target.endsWith(")V")) {
            throw new ClassFormatError(
                    "delegation requires void target and delegate: "
                            + target + " -> " + delegateName + delegateDescriptor);
        }
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        return patcher.patch(
                java.util.Collections.singleton(target), delegateName, delegateDescriptor);
    }

    private byte[] patch(
            java.util.Set<String> targets, String delegateName, String delegateDescriptor) {
        String[] pool = readHeaderAndConstantPool();

        skip(2); // access_flags
        int thisClass = readU2();
        if (delegateName != null) {
            delegateRef = findSelfMethodRef(pool, thisClass, delegateName, delegateDescriptor);
        }
        skip(2); // super_class
        int interfaceCount = readU2();
        skip(interfaceCount * 2);

        skipMembers(); // fields

        int methodCount = readU2();
        // Collected in file order; applied last-to-first so that an earlier
        // edit cannot shift the offsets recorded for a later one.
        java.util.List<Edit> edits = new java.util.ArrayList<Edit>();
        for (int i = 0; i < methodCount; i++) {
            Edit edit = scanMethod(pool, targets);
            if (edit != null) {
                edits.add(edit);
            }
        }

        if (edits.isEmpty()) {
            return null;
        }
        return applyEdits(edits);
    }

    /** Reads magic, version and the constant pool; returns UTF-8 entries by index. */
    private String[] readHeaderAndConstantPool() {
        int magic = readU4();
        if (magic != 0xCAFEBABE) {
            throw new ClassFormatError("not a class file: magic=0x" + Integer.toHexString(magic));
        }
        skip(2); // minor_version
        skip(2); // major_version

        int poolCount = readU2();
        String[] pool = new String[poolCount];
        tags = new int[poolCount];
        operandA = new int[poolCount];
        operandB = new int[poolCount];
        // Index 0 is unused; long/double consume two slots (JVMS 4.4.5).
        for (int i = 1; i < poolCount; i++) {
            int tag = readU1();
            tags[i] = tag;
            switch (tag) {
                case CONSTANT_UTF8:
                    int length = readU2();
                    pool[i] = new String(buf, pos, length, java.nio.charset.StandardCharsets.UTF_8);
                    skip(length);
                    break;
                case CONSTANT_METHODREF:
                case CONSTANT_NAME_AND_TYPE:
                    // The two shapes delegation resolves through: a Methodref
                    // is (class_index, name_and_type_index), a NameAndType is
                    // (name_index, descriptor_index).
                    operandA[i] = readU2();
                    operandB[i] = readU2();
                    break;
                case CONSTANT_INTEGER:
                case CONSTANT_FLOAT:
                case CONSTANT_FIELDREF:
                case CONSTANT_INTERFACE_METHODREF:
                case CONSTANT_DYNAMIC:
                case CONSTANT_INVOKE_DYNAMIC:
                    skip(4);
                    break;
                case CONSTANT_LONG:
                case CONSTANT_DOUBLE:
                    skip(8);
                    i++;
                    break;
                case CONSTANT_CLASS:
                    operandA[i] = readU2(); // name_index
                    break;
                case CONSTANT_STRING:
                case CONSTANT_METHOD_TYPE:
                case CONSTANT_MODULE:
                case CONSTANT_PACKAGE:
                    skip(2);
                    break;
                case CONSTANT_METHOD_HANDLE:
                    skip(3);
                    break;
                default:
                    throw new ClassFormatError("unknown constant pool tag " + tag + " at index " + i);
            }
        }
        return pool;
    }

    /**
     * Finds the pool index of a Methodref on this class itself.
     *
     * <p>Matched by class <em>name</em> rather than by index identity: a class
     * file may legally carry duplicate Class entries naming the same type, and
     * only the name says which type a Methodref really targets.
     */
    private int findSelfMethodRef(String[] pool, int thisClass, String name, String descriptor) {
        String selfName = pool[operandA[thisClass]];
        for (int i = 1; i < tags.length; i++) {
            if (tags[i] != CONSTANT_METHODREF) {
                continue;
            }
            int nat = operandB[i];
            if (pool[operandA[operandA[i]]].equals(selfName)
                    && name.equals(pool[operandA[nat]])
                    && descriptor.equals(pool[operandB[nat]])) {
                return i;
            }
        }
        throw new ClassFormatError(
                "no Methodref for " + selfName + "." + name + descriptor
                        + " in the pool -- delegation forges nothing, so the class must"
                        + " already call the delegate on itself somewhere");
    }

    /** Skips a fields_count/methods_count-prefixed member table. */
    private void skipMembers() {
        int count = readU2();
        for (int i = 0; i < count; i++) {
            skip(6); // access_flags, name_index, descriptor_index
            skipAttributes();
        }
    }

    private void skipAttributes() {
        int count = readU2();
        for (int i = 0; i < count; i++) {
            skip(2); // attribute_name_index
            int length = readU4();
            skip(length);
        }
    }

    /**
     * Parses one method_info. Returns an {@link Edit} if it is a patch target
     * and carries a Code attribute, otherwise null.
     */
    private Edit scanMethod(String[] pool, java.util.Set<String> targets) {
        int accessFlags = readU2();
        String name = pool[readU2()];
        String descriptor = pool[readU2()];
        boolean wanted = targets.contains(name) || targets.contains(name + descriptor);

        Edit edit = null;
        int attributeCount = readU2();
        for (int i = 0; i < attributeCount; i++) {
            String attributeName = pool[readU2()];
            int lengthOffset = pos;
            int length = readU4();
            if (wanted && "Code".equals(attributeName)) {
                if (delegateRef >= 0 && (accessFlags & ACC_STATIC) != 0) {
                    throw new ClassFormatError(
                            "cannot delegate static " + name + descriptor + " through this");
                }
                // Replace from the attribute_length field through the end of
                // the attribute body. StackMapTable, LineNumberTable and the
                // exception table live inside Code and go with it -- which is
                // exactly right, since neither a no-op nor a straight-line
                // self-call has a branch to map.
                byte[] body =
                        delegateRef >= 0
                                ? buildDelegateCodeAttribute(descriptor, accessFlags, delegateRef)
                                : buildCodeAttribute(descriptor, accessFlags);
                edit = new Edit(lengthOffset, pos + length, body);
            }
            skip(length);
        }
        return edit;
    }

    /**
     * Builds a complete Code attribute body (attribute_length included) holding
     * the smallest legal body that satisfies {@code descriptor}'s return type.
     */
    private static byte[] buildCodeAttribute(String descriptor, int accessFlags) {
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
     * Builds a complete Code attribute whose body is {@code this.<delegate>();
     * return}. Straight-line, so no StackMapTable is needed at any class file
     * version; one reference on the stack, so max_stack is one.
     */
    private static byte[] buildDelegateCodeAttribute(
            String descriptor, int accessFlags, int methodRef) {
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
                throw new ClassFormatError("unsupported return type '" + returnType + "' in " + descriptor);
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

    /** Splices every edit into a fresh buffer, applying last-to-first so offsets stay valid. */
    private byte[] applyEdits(java.util.List<Edit> edits) {
        byte[] result = buf;
        for (int i = edits.size() - 1; i >= 0; i--) {
            Edit edit = edits.get(i);
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            out.write(result, 0, edit.start);
            out.write(edit.replacement, 0, edit.replacement.length);
            out.write(result, edit.end, result.length - edit.end);
            result = out.toByteArray();
        }
        return result;
    }

    private int readU1() {
        return buf[pos++] & 0xff;
    }

    private int readU2() {
        return (readU1() << 8) | readU1();
    }

    private int readU4() {
        return (readU2() << 16) | readU2();
    }

    private void skip(int count) {
        pos += count;
    }

    private static void writeU2(java.io.ByteArrayOutputStream out, int value) {
        out.write((value >>> 8) & 0xff);
        out.write(value & 0xff);
    }

    private static void writeU4(java.io.ByteArrayOutputStream out, int value) {
        out.write((value >>> 24) & 0xff);
        out.write((value >>> 16) & 0xff);
        out.write((value >>> 8) & 0xff);
        out.write(value & 0xff);
    }

    /** A byte range in the original class file and the bytes replacing it. */
    private static final class Edit {
        final int start;
        final int end;
        final byte[] replacement;

        Edit(int start, int end, byte[] replacement) {
            this.start = start;
            this.end = end;
            this.replacement = replacement;
        }
    }
}
