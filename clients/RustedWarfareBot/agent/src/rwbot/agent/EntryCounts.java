package rwbot.agent;

/**
 * Prepends an {@code invokestatic} entry counter to named methods of a
 * loaded class -- the wall-decoupling program's first instrument (wiki:
 * policy-determinism, dettwin96).
 *
 * <p>Split from {@link ClassFilePatcher} at the module ceiling like
 * {@link Bytecode} and {@link CodeBodies} before it, and along the same
 * boundary: the patcher owns the parse-and-splice plumbing every patch
 * family shares (the pool reader, the member walk, {@link Edit} and its
 * last-to-first application), while this module owns one family's edit
 * arithmetic -- what a four-byte prepend shifts and how each shifted
 * structure is fixed.
 *
 * <p>FOUR bytes, deliberately, not three: {@code tableswitch} and
 * {@code lookupswitch} pad to a four-byte boundary measured from the start
 * of the code array, so a four-byte prepend ({@code invokestatic} plus
 * {@code nop}) preserves every switch's padding and the whole body shifts
 * uniformly -- which is what keeps every RELATIVE branch offset valid
 * without rewriting one. What the shift does reach is fixed explicitly:
 * the exception table's three pc columns, and the FIRST StackMapTable
 * frame's offset delta (later frames are deltas from their predecessor and
 * shift for free). A first frame whose new delta no longer fits its
 * compact tag is re-encoded to the extended form, growing the attribute by
 * two bytes, and both length fields above it grow to match. Debug tables
 * (LineNumberTable, LocalVariableTable) are left untouched: the only
 * misattribution is the injected invoke inheriting the first real
 * instruction's line, which is where it conceptually belongs. The stack
 * shape is untouched by construction -- a {@code ()V} static call pushes
 * and pops nothing, so {@code max_stack} and {@code max_locals} stand.
 */
final class EntryCounts {

    private EntryCounts() {
    }

    /**
     * Returns a copy of {@code classFile} in which every method named in
     * {@code counters} (keyed {@code name + descriptor}, e.g.
     * {@code "f()V"}) begins with
     * {@code invokestatic counterOwner.<value>()V} followed by a
     * {@code nop}.
     *
     * @param classFile The class bytes as loaded.
     * @param counters Target methods to counter methods on the owner, in a
     *     deterministic order (the pool layout follows it).
     * @param counterOwner Internal name of the class holding the counters,
     *     e.g. {@code rwbot/agent/ThinkCount}.
     * @return The patched bytes.
     * @throws ClassFormatError if the class cannot be parsed, any requested
     *     target has no Code attribute in this class, a patched body would
     *     pass the 65535-byte code limit, or the pool cannot grow -- every
     *     one means the pinned build moved, and a loud stop at class load
     *     beats a diagnostic silently counting nothing.
     */
    static byte[] prepend(
            byte[] classFile,
            java.util.LinkedHashMap<String, String> counters,
            String counterOwner) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        String[] pool = patcher.readHeaderAndConstantPool();
        int poolEnd = patcher.pos;
        int poolCount = patcher.tags.length;
        int appendedEntries = 3 + 3 * counters.size();
        if (poolCount + appendedEntries > 0xffff) {
            throw new ClassFormatError("constant pool too large to grow: " + poolCount);
        }
        // Layout: owner Utf8, owner Class, "()V" Utf8, then per counter a
        // name Utf8, a NameAndType over (name, "()V"), and a Methodref over
        // (owner Class, NameAndType) -- descriptor and owner shared so the
        // callee must match by construction, the retarget's own rule.
        int ownerClass = poolCount + 1;
        int voidDescriptor = poolCount + 2;
        java.io.ByteArrayOutputStream appended = new java.io.ByteArrayOutputStream();
        byte[] ownerUtf8 = counterOwner.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        appended.write(ClassFilePatcher.CONSTANT_UTF8);
        CodeBodies.writeU2(appended, ownerUtf8.length);
        appended.write(ownerUtf8, 0, ownerUtf8.length);
        appended.write(ClassFilePatcher.CONSTANT_CLASS);
        CodeBodies.writeU2(appended, poolCount);
        appended.write(ClassFilePatcher.CONSTANT_UTF8);
        CodeBodies.writeU2(appended, 3);
        appended.write('(');
        appended.write(')');
        appended.write('V');
        java.util.Map<String, Integer> refOf = new java.util.LinkedHashMap<String, Integer>();
        int next = poolCount + 3;
        for (java.util.Map.Entry<String, String> counter : counters.entrySet()) {
            byte[] nameUtf8 =
                    counter.getValue().getBytes(java.nio.charset.StandardCharsets.UTF_8);
            appended.write(ClassFilePatcher.CONSTANT_UTF8);
            CodeBodies.writeU2(appended, nameUtf8.length);
            appended.write(nameUtf8, 0, nameUtf8.length);
            appended.write(ClassFilePatcher.CONSTANT_NAME_AND_TYPE);
            CodeBodies.writeU2(appended, next);
            CodeBodies.writeU2(appended, voidDescriptor);
            appended.write(ClassFilePatcher.CONSTANT_METHODREF);
            CodeBodies.writeU2(appended, ownerClass);
            CodeBodies.writeU2(appended, next + 1);
            refOf.put(counter.getKey(), Integer.valueOf(next + 2));
            next += 3;
        }

        patcher.skip(2); // access_flags
        patcher.skip(2); // this_class
        patcher.skip(2); // super_class
        int interfaceCount = patcher.readU2();
        patcher.skip(interfaceCount * 2);
        patcher.skipMembers(); // fields

        java.util.List<Edit> edits = new java.util.ArrayList<Edit>();
        int newCount = poolCount + appendedEntries;
        byte[] countPatch = {(byte) ((newCount >>> 8) & 0xff), (byte) (newCount & 0xff)};
        edits.add(new Edit(8, 10, countPatch));
        edits.add(new Edit(poolEnd, poolEnd, appended.toByteArray()));

        java.util.Set<String> unmatched = new java.util.LinkedHashSet<String>(counters.keySet());
        int methodCount = patcher.readU2();
        for (int i = 0; i < methodCount; i++) {
            collect(patcher, pool, refOf, unmatched, edits);
        }
        if (!unmatched.isEmpty()) {
            throw new ClassFormatError(
                    "no Code attribute matched entry-count target(s) " + unmatched
                            + " -- the pinned engine build moved under the instrument");
        }
        return patcher.applyEdits(edits);
    }

    /**
     * Parses one method_info; when it is an entry-count target, records the
     * prepend and every fixup its shift requires.
     */
    private static void collect(
            ClassFilePatcher patcher,
            String[] pool,
            java.util.Map<String, Integer> refOf,
            java.util.Set<String> unmatched,
            java.util.List<Edit> edits) {
        patcher.skip(2); // access_flags
        String name = pool[patcher.readU2()];
        String descriptor = pool[patcher.readU2()];
        Integer ref = refOf.get(name + descriptor);
        int attributeCount = patcher.readU2();
        for (int i = 0; i < attributeCount; i++) {
            String attributeName = pool[patcher.readU2()];
            int lengthOffset = patcher.pos;
            int length = patcher.readU4();
            if (ref != null && "Code".equals(attributeName)) {
                unmatched.remove(name + descriptor);
                emit(patcher.buf, pool, ref.intValue(), lengthOffset, length, edits);
            }
            patcher.skip(length);
        }
    }

    /**
     * Records the edits for one target Code attribute: both length fields,
     * the four injected bytes, the exception table's shifted pcs, and the
     * first StackMapTable frame's shifted delta (re-encoded to the extended
     * form when the compact tag can no longer carry it).
     */
    private static void emit(
            byte[] buf,
            String[] pool,
            int ref,
            int lengthOffset,
            int length,
            java.util.List<Edit> edits) {
        int codeLengthPos = lengthOffset + 4 + 4; // max_stack, max_locals.
        int codeLength = readU4At(buf, codeLengthPos);
        if (codeLength + 4 > 0xffff) {
            throw new ClassFormatError(
                    "entry counter would pass the 65535-byte code limit at " + codeLength);
        }
        int codeStart = codeLengthPos + 4;
        byte[] entry = {(byte) 0xb8, (byte) ((ref >>> 8) & 0xff), (byte) (ref & 0xff), 0x00};

        int exceptionCountPos = codeStart + codeLength;
        int exceptionCount = readU2At(buf, exceptionCountPos);
        int exceptionStart = exceptionCountPos + 2;
        Edit exceptionEdit = null;
        if (exceptionCount > 0) {
            byte[] fixed = new byte[exceptionCount * 8];
            System.arraycopy(buf, exceptionStart, fixed, 0, fixed.length);
            for (int e = 0; e < exceptionCount; e++) {
                for (int column = 0; column < 3; column++) { // start, end, handler.
                    int at = e * 8 + column * 2;
                    int pc = ((fixed[at] & 0xff) << 8 | (fixed[at + 1] & 0xff)) + 4;
                    fixed[at] = (byte) ((pc >>> 8) & 0xff);
                    fixed[at + 1] = (byte) (pc & 0xff);
                }
            }
            exceptionEdit = new Edit(exceptionStart, exceptionStart + fixed.length, fixed);
        }

        int growth = 0;
        Edit frameEdit = null;
        Edit frameLengthEdit = null;
        int innerCountPos = exceptionStart + exceptionCount * 8;
        int innerCount = readU2At(buf, innerCountPos);
        int at = innerCountPos + 2;
        for (int a = 0; a < innerCount; a++) {
            int attrName = readU2At(buf, at);
            int attrLength = readU4At(buf, at + 2);
            if ("StackMapTable".equals(pool[attrName]) && readU2At(buf, at + 6) > 0) {
                int framePos = at + 8;
                byte[] head = shiftedFrameHead(buf, framePos);
                int oldHead = (buf[framePos] & 0xff) <= 127 ? 1 : 3;
                growth = head.length - oldHead;
                frameEdit = new Edit(framePos, framePos + oldHead, head);
                if (growth != 0) {
                    frameLengthEdit = new Edit(at + 2, at + 6, u4(attrLength + growth));
                }
            }
            at += 6 + attrLength;
        }

        edits.add(new Edit(lengthOffset, lengthOffset + 4, u4(length + 4 + growth)));
        edits.add(new Edit(codeLengthPos, codeLengthPos + 4, u4(codeLength + 4)));
        edits.add(new Edit(codeStart, codeStart, entry));
        if (exceptionEdit != null) {
            edits.add(exceptionEdit);
        }
        if (frameLengthEdit != null) {
            edits.add(frameLengthEdit);
        }
        if (frameEdit != null) {
            edits.add(frameEdit);
        }
    }

    /**
     * Renders the first StackMapTable frame's head with its offset delta
     * grown by four -- the uniform shift the prepend applies to every
     * bytecode offset. Only the FIRST frame carries an absolute delta;
     * every later frame is a delta from its predecessor and needs nothing.
     *
     * @param buf The class bytes.
     * @param framePos Offset of the first frame's tag byte.
     * @return The replacement head: same tag with the bigger compact delta,
     *     the delta bumped in place for the u2 forms, or the extended
     *     re-encoding when the compact form can no longer carry it.
     */
    private static byte[] shiftedFrameHead(byte[] buf, int framePos) {
        int tag = buf[framePos] & 0xff;
        if (tag <= 63) {
            int delta = tag + 4;
            if (delta <= 63) {
                return new byte[] {(byte) delta};
            }
            return new byte[] {(byte) 251, (byte) ((delta >>> 8) & 0xff), (byte) (delta & 0xff)};
        }
        if (tag <= 127) {
            int delta = tag - 64 + 4;
            if (delta <= 63) {
                return new byte[] {(byte) (64 + delta)};
            }
            return new byte[] {(byte) 247, (byte) ((delta >>> 8) & 0xff), (byte) (delta & 0xff)};
        }
        if (tag < 247) {
            // 128-246 are reserved (JVMS 4.7.4); a file carrying one is not
            // a frame this shift understands, and guessing would corrupt it.
            throw new ClassFormatError("reserved StackMapTable frame tag " + tag);
        }
        int delta = readU2At(buf, framePos + 1) + 4;
        return new byte[] {(byte) tag, (byte) ((delta >>> 8) & 0xff), (byte) (delta & 0xff)};
    }

    private static byte[] u4(int value) {
        return new byte[] {
            (byte) ((value >>> 24) & 0xff),
            (byte) ((value >>> 16) & 0xff),
            (byte) ((value >>> 8) & 0xff),
            (byte) (value & 0xff),
        };
    }

    private static int readU2At(byte[] buf, int at) {
        return (buf[at] & 0xff) << 8 | (buf[at + 1] & 0xff);
    }

    private static int readU4At(byte[] buf, int at) {
        return (readU2At(buf, at) << 16) | readU2At(buf, at + 2);
    }
}
