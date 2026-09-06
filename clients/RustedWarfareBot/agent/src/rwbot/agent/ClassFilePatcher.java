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

    /**
     * Returns a copy of {@code classFile} in which, inside every method named
     * in {@code targets}, each {@code invokestatic} of
     * {@code fromOwner.name:descriptor} is retargeted to
     * {@code toOwner.name:descriptor}; {@code null} if the class holds no
     * Methodref for the source call (nothing to rewrite) or no target method
     * invokes it.
     *
     * <p>This is the one edit here that grows the constant pool -- by exactly
     * three entries (a Utf8 for {@code toOwner}, a Class over it, and a
     * Methodref pairing that Class with the ORIGINAL call's NameAndType, so
     * name and descriptor are shared rather than duplicated and the callee
     * must match the caller's expectations by construction). The pool count
     * is a fixed-offset u2 and the entries append at the pool's end, so the
     * rest of the file is untouched except the two operand bytes of each
     * rewritten invoke -- same instruction length, so no attribute length
     * anywhere changes.
     *
     * <p>The rewrite is scoped to the NAMED methods on purpose: other methods
     * of the same class may draw through the same helper for simulation
     * purposes, and a class-wide rewrite would move draws nobody audited
     * (wiki: policy-determinism, the 2026-09-06 arc).
     *
     * @throws ClassFormatError if the class cannot be parsed, or a targeted
     *     method's bytecode contains an instruction the walker does not
     *     model -- a loud stop at class load beats a mis-parsed operand
     *     silently rewriting the wrong bytes.
     */
    static byte[] retargetStaticInvokes(
            byte[] classFile,
            java.util.Set<String> targets,
            String fromOwner,
            String name,
            String descriptor,
            String toOwner) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        return patcher.retarget(targets, null, fromOwner, name, descriptor, toOwner);
    }

    /**
     * The line-scoped variant: rewrites only the invokes whose bytecode
     * offsets the target method's own {@code LineNumberTable} attributes to
     * one of {@code lines}.
     *
     * <p>Method scope is too coarse exactly once so far, and it is the case
     * that motivated this: the data-driven unit class's per-tick update holds
     * TWELVE {@code f.c(FF)F} invokes, of which three were bytecode-verified
     * as effect-only (an angle jitter feeding a particle spawn and two
     * velocity jitters written to the particle's own fields) while the other
     * nine sit on unread paths that may be simulation. The draw tap names a
     * site by its source line, the lawfulness read is performed against that
     * line's bytecode, and this filter consumes the SAME line number -- one
     * identifier from measurement to patch, resolved through the pinned
     * jar's own line table rather than through offsets someone once copied
     * (wiki: policy-determinism, the 2026-09-06 arc).
     *
     * @throws ClassFormatError if any requested line matched no rewritten
     *     invoke -- a silent partial patch would measure as the very noise
     *     this exists to remove -- or the method carries no line table to
     *     resolve against.
     */
    static byte[] retargetStaticInvokesAtLines(
            byte[] classFile,
            String targetMethod,
            java.util.Set<Integer> lines,
            String fromOwner,
            String name,
            String descriptor,
            String toOwner) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        byte[] result =
                patcher.retarget(
                        java.util.Collections.singleton(targetMethod),
                        lines,
                        fromOwner,
                        name,
                        descriptor,
                        toOwner);
        if (result != null && !patcher.unmatchedLines.isEmpty()) {
            throw new ClassFormatError(
                    "requested lines matched no " + fromOwner + "." + name + descriptor
                            + " invoke in " + targetMethod + ": " + patcher.unmatchedLines
                            + " -- the pinned jar's line table moved, or the read was wrong");
        }
        return result;
    }

    // Lines the caller asked for that no rewritten invoke resolved to;
    // populated by the line-scoped walk so the entry point can refuse a
    // partial patch loudly.
    private final java.util.Set<Integer> unmatchedLines =
            new java.util.LinkedHashSet<Integer>();

    private byte[] retarget(
            java.util.Set<String> targets,
            java.util.Set<Integer> lines,
            String fromOwner,
            String name,
            String descriptor,
            String toOwner) {
        String[] pool = readHeaderAndConstantPool();
        int poolEnd = pos;
        int poolCount = tags.length;

        int fromRef = -1;
        for (int i = 1; i < poolCount; i++) {
            if (tags[i] != CONSTANT_METHODREF) {
                continue;
            }
            int nat = operandB[i];
            if (pool[operandA[operandA[i]]].equals(fromOwner)
                    && name.equals(pool[operandA[nat]])
                    && descriptor.equals(pool[operandB[nat]])) {
                fromRef = i;
                break;
            }
        }
        if (fromRef < 0) {
            return null;
        }
        if (poolCount + 3 > 0xffff) {
            // The three appended indices must fit a u2; past this the writes
            // would truncate into a silently corrupt pool.
            throw new ClassFormatError("constant pool too large to grow: " + poolCount);
        }
        int newRef = poolCount + 2; // Utf8 at poolCount, Class at +1, Methodref at +2.

        skip(2); // access_flags
        skip(2); // this_class
        skip(2); // super_class
        int interfaceCount = readU2();
        skip(interfaceCount * 2);
        skipMembers(); // fields

        if (lines != null) {
            unmatchedLines.addAll(lines);
        }
        java.util.List<Edit> edits = new java.util.ArrayList<Edit>();
        int methodCount = readU2();
        for (int i = 0; i < methodCount; i++) {
            collectInvokeEdits(pool, targets, lines, fromRef, newRef, edits);
        }
        if (edits.isEmpty()) {
            return null;
        }

        // The pool grows only when something was actually rewritten, and the
        // count patch plus the appended entries go FIRST in the edit list so
        // applyEdits' last-to-first order splices the later (larger-offset)
        // operand rewrites before the insertion shifts anything.
        byte[] countPatch = {(byte) (((poolCount + 3) >>> 8) & 0xff), (byte) ((poolCount + 3) & 0xff)};
        edits.add(0, new Edit(8, 10, countPatch));
        byte[] toOwnerUtf8 = toOwner.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        java.io.ByteArrayOutputStream appended = new java.io.ByteArrayOutputStream();
        appended.write(CONSTANT_UTF8);
        CodeBodies.writeU2(appended, toOwnerUtf8.length);
        appended.write(toOwnerUtf8, 0, toOwnerUtf8.length);
        appended.write(CONSTANT_CLASS);
        CodeBodies.writeU2(appended, poolCount);
        appended.write(CONSTANT_METHODREF);
        CodeBodies.writeU2(appended, poolCount + 1);
        CodeBodies.writeU2(appended, operandB[fromRef]);
        edits.add(1, new Edit(poolEnd, poolEnd, appended.toByteArray()));
        return applyEdits(edits);
    }

    /**
     * Parses one method_info; when it is a retarget target, walks its Code
     * attribute and records a two-byte operand edit for every
     * {@code invokestatic fromRef}.
     */
    private void collectInvokeEdits(
            String[] pool,
            java.util.Set<String> targets,
            java.util.Set<Integer> lines,
            int fromRef,
            int newRef,
            java.util.List<Edit> edits) {
        skip(2); // access_flags
        String name = pool[readU2()];
        String descriptor = pool[readU2()];
        boolean wanted = targets.contains(name) || targets.contains(name + descriptor);

        int attributeCount = readU2();
        for (int i = 0; i < attributeCount; i++) {
            String attributeName = pool[readU2()];
            int length = readU4();
            if (wanted && "Code".equals(attributeName)) {
                int codeLength = ((buf[pos + 4] & 0xff) << 24)
                        | ((buf[pos + 5] & 0xff) << 16)
                        | ((buf[pos + 6] & 0xff) << 8)
                        | (buf[pos + 7] & 0xff);
                int codeStart = pos + 8; // max_stack, max_locals, code_length.
                int[] lineStarts = null;
                int[] lineNumbers = null;
                if (lines != null) {
                    int[][] table = Bytecode.readLineTable(buf, pool, codeStart, codeLength);
                    lineStarts = table[0];
                    lineNumbers = table[1];
                    if (lineStarts.length == 0) {
                        throw new ClassFormatError(
                                "no LineNumberTable in " + name + descriptor
                                        + " to resolve a line-scoped retarget against");
                    }
                }
                byte[] operand = {(byte) ((newRef >>> 8) & 0xff), (byte) (newRef & 0xff)};
                int at = 0;
                while (at < codeLength) {
                    int opcode = buf[codeStart + at] & 0xff;
                    if (opcode == 0xb8 // invokestatic
                            && ((buf[codeStart + at + 1] & 0xff) << 8 | (buf[codeStart + at + 2] & 0xff))
                                    == fromRef
                            && (lines == null
                                    || matchesLine(lines, lineStarts, lineNumbers, at))) {
                        edits.add(new Edit(codeStart + at + 1, codeStart + at + 3, operand));
                    }
                    at += Bytecode.instructionLength(buf, opcode, codeStart, at);
                }
            }
            skip(length);
        }
    }

    /**
     * Whether the instruction at {@code at} belongs to one of the requested
     * source lines. A hit is also consumed from {@link #unmatchedLines},
     * which is how the entry point knows every requested line did real work.
     */
    private boolean matchesLine(
            java.util.Set<Integer> lines, int[] lineStarts, int[] lineNumbers, int at) {
        Integer line = Integer.valueOf(Bytecode.lineAt(lineStarts, lineNumbers, at));
        if (lines.contains(line)) {
            unmatchedLines.remove(line);
            return true;
        }
        return false;
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
                                ? CodeBodies.delegate(descriptor, accessFlags, delegateRef)
                                : CodeBodies.noOp(descriptor, accessFlags);
                edit = new Edit(lengthOffset, pos + length, body);
            }
            skip(length);
        }
        return edit;
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
