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
 * <p>A MULTIPLE OF FOUR bytes, deliberately: {@code tableswitch} and
 * {@code lookupswitch} pad to a four-byte boundary measured from the start
 * of the code array, so a four-byte prepend ({@code invokestatic} plus
 * {@code nop}, or {@code aload_0} plus {@code invokestatic}) -- or the
 * value shape's eight ({@code aload_0}, {@code fload_1},
 * {@code invokestatic}, three {@code nop}) -- preserves every switch's
 * padding and the whole body shifts uniformly, which is what keeps every
 * RELATIVE branch offset valid
 * without rewriting one. What the shift does reach is fixed explicitly:
 * the exception table's three pc columns, and the FIRST StackMapTable
 * frame's offset delta (later frames are deltas from their predecessor and
 * shift for free). A first frame whose new delta no longer fits its
 * compact tag is re-encoded to the extended form, growing the attribute by
 * two bytes, and both length fields above it grow to match. The
 * LineNumberTable shifts with the code -- reported lines are the tap's
 * measurement language, so a drifted table is a lying instrument
 * ({@link #shiftedLineTable}) -- while LocalVariableTable stays untouched:
 * nothing in this stack reads it, and the verifier ignores it. The stack
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
     * {@code nop}, and every method named in {@code receiverHooks} begins
     * with {@code aload_0; invokestatic counterOwner.<value>(Object)V} --
     * also four bytes, so the same alignment argument covers both shapes,
     * and the hook receives the instance whose state the scan is about to
     * read ({@link ThinkCount#scan}).
     *
     * @param classFile The class bytes as loaded.
     * @param counters Target methods to no-argument counter methods on the
     *     owner, in a deterministic order (the pool layout follows it).
     * @param receiverHooks Target INSTANCE methods to
     *     {@code (Ljava/lang/Object;)V} hook methods on the owner. The
     *     receiver push makes this shape wrong for a static target, and
     *     {@code max_stack} is raised to one where a body of zero would
     *     otherwise underprovision the push.
     * @param valueHooks Target INSTANCE methods whose FIRST parameter is a
     *     float, to {@code (Ljava/lang/Object;F)V} hook methods on the
     *     owner -- the eight-byte shape, receiver and argument both in
     *     hand, for hooks that must judge the value (the spend trace).
     *     {@code max_stack} is raised to two where the body provisions
     *     less.
     * @param counterOwner Internal name of the class holding the hooks,
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
            java.util.LinkedHashMap<String, String> receiverHooks,
            java.util.LinkedHashMap<String, String> valueHooks,
            String counterOwner) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        String[] pool = patcher.readHeaderAndConstantPool();
        int poolEnd = patcher.pos;
        int poolCount = patcher.tags.length;
        int appendedEntries =
                3
                        + 3 * counters.size()
                        + (receiverHooks.isEmpty() ? 0 : 1)
                        + 3 * receiverHooks.size()
                        + (valueHooks.isEmpty() ? 0 : 1)
                        + 3 * valueHooks.size();
        if (poolCount + appendedEntries > 0xffff) {
            throw new ClassFormatError("constant pool too large to grow: " + poolCount);
        }
        // Layout: owner Utf8, owner Class, "()V" Utf8, then per counter a
        // name Utf8, a NameAndType over (name, "()V"), and a Methodref over
        // (owner Class, NameAndType) -- descriptor and owner shared so the
        // callee must match by construction, the retarget's own rule. The
        // receiver hooks follow with their own shared "(Ljava/lang/Object;)V".
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
        java.util.Set<String> withReceiver = new java.util.LinkedHashSet<String>();
        int next = poolCount + 3;
        for (java.util.Map.Entry<String, String> counter : counters.entrySet()) {
            next = appendHook(appended, counter.getValue(), ownerClass, voidDescriptor, next);
            refOf.put(counter.getKey(), Integer.valueOf(next - 1));
        }
        if (!receiverHooks.isEmpty()) {
            byte[] objectDescriptor =
                    "(Ljava/lang/Object;)V".getBytes(java.nio.charset.StandardCharsets.UTF_8);
            appended.write(ClassFilePatcher.CONSTANT_UTF8);
            CodeBodies.writeU2(appended, objectDescriptor.length);
            appended.write(objectDescriptor, 0, objectDescriptor.length);
            int receiverDescriptor = next;
            next += 1;
            for (java.util.Map.Entry<String, String> hook : receiverHooks.entrySet()) {
                next = appendHook(appended, hook.getValue(), ownerClass, receiverDescriptor, next);
                refOf.put(hook.getKey(), Integer.valueOf(next - 1));
                withReceiver.add(hook.getKey());
            }
        }
        java.util.Set<String> withValue = new java.util.LinkedHashSet<String>();
        if (!valueHooks.isEmpty()) {
            byte[] valueDescriptor =
                    "(Ljava/lang/Object;F)V".getBytes(java.nio.charset.StandardCharsets.UTF_8);
            appended.write(ClassFilePatcher.CONSTANT_UTF8);
            CodeBodies.writeU2(appended, valueDescriptor.length);
            appended.write(valueDescriptor, 0, valueDescriptor.length);
            int descriptorIndex = next;
            next += 1;
            for (java.util.Map.Entry<String, String> hook : valueHooks.entrySet()) {
                next = appendHook(appended, hook.getValue(), ownerClass, descriptorIndex, next);
                refOf.put(hook.getKey(), Integer.valueOf(next - 1));
                withValue.add(hook.getKey());
            }
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

        java.util.Set<String> unmatched = new java.util.LinkedHashSet<String>(refOf.keySet());
        int methodCount = patcher.readU2();
        for (int i = 0; i < methodCount; i++) {
            collect(patcher, pool, refOf, withReceiver, withValue, unmatched, edits);
        }
        if (!unmatched.isEmpty()) {
            throw new ClassFormatError(
                    "no Code attribute matched entry-count target(s) " + unmatched
                            + " -- the pinned engine build moved under the instrument");
        }
        return patcher.applyEdits(edits);
    }

    /**
     * Appends one hook's pool triple -- name Utf8, NameAndType over the
     * shared descriptor, Methodref over the shared owner Class -- and
     * returns the next free pool index. The Methodref lands at the returned
     * index minus one.
     */
    private static int appendHook(
            java.io.ByteArrayOutputStream appended,
            String hookName,
            int ownerClass,
            int descriptor,
            int next) {
        byte[] nameUtf8 = hookName.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        appended.write(ClassFilePatcher.CONSTANT_UTF8);
        CodeBodies.writeU2(appended, nameUtf8.length);
        appended.write(nameUtf8, 0, nameUtf8.length);
        appended.write(ClassFilePatcher.CONSTANT_NAME_AND_TYPE);
        CodeBodies.writeU2(appended, next);
        CodeBodies.writeU2(appended, descriptor);
        appended.write(ClassFilePatcher.CONSTANT_METHODREF);
        CodeBodies.writeU2(appended, ownerClass);
        CodeBodies.writeU2(appended, next + 1);
        return next + 3;
    }

    /**
     * Parses one method_info; when it is an entry-count target, records the
     * prepend and every fixup its shift requires.
     */
    private static void collect(
            ClassFilePatcher patcher,
            String[] pool,
            java.util.Map<String, Integer> refOf,
            java.util.Set<String> withReceiver,
            java.util.Set<String> withValue,
            java.util.Set<String> unmatched,
            java.util.List<Edit> edits) {
        patcher.skip(2); // access_flags
        String name = pool[patcher.readU2()];
        String descriptor = pool[patcher.readU2()];
        String key = name + descriptor;
        Integer ref = refOf.get(key);
        int attributeCount = patcher.readU2();
        for (int i = 0; i < attributeCount; i++) {
            String attributeName = pool[patcher.readU2()];
            int lengthOffset = patcher.pos;
            int length = patcher.readU4();
            if (ref != null && "Code".equals(attributeName)) {
                unmatched.remove(key);
                int hi = (ref.intValue() >>> 8) & 0xff;
                int lo = ref.intValue() & 0xff;
                byte[] entry;
                int minStack;
                if (withValue.contains(key)) {
                    // 0x23 is fload_1 -- the float argument's slot in an
                    // instance method. 0x22 (fload_0) loads the RECEIVER
                    // slot as a float, which the selftest's define+resolve
                    // did NOT reject; the live link did, at match boot
                    // (wiki log 2026-09-11) -- a verifier-green selftest
                    // bounds parse errors, not type errors.
                    entry =
                            new byte[] {
                                0x2a, 0x23, (byte) 0xb8, (byte) hi, (byte) lo, 0x00, 0x00, 0x00
                            };
                    minStack = 2;
                } else if (withReceiver.contains(key)) {
                    entry = new byte[] {0x2a, (byte) 0xb8, (byte) hi, (byte) lo};
                    minStack = 1;
                } else {
                    entry = new byte[] {(byte) 0xb8, (byte) hi, (byte) lo, 0x00};
                    minStack = 0;
                }
                emit(patcher.buf, pool, entry, minStack, lengthOffset, length, edits);
            }
            patcher.skip(length);
        }
    }

    /**
     * Records the edits for one target Code attribute: both length fields,
     * the injected entry bytes, the exception table's shifted pcs, the
     * LineNumberTable's shifted pcs, and the first StackMapTable frame's
     * shifted delta (re-encoded to the extended form when the compact tag
     * can no longer carry it). Every shift is the entry's own length -- a
     * multiple of four by construction of the shapes in {@code collect}.
     */
    private static void emit(
            byte[] buf,
            String[] pool,
            byte[] entry,
            int minStack,
            int lengthOffset,
            int length,
            java.util.List<Edit> edits) {
        int shift = entry.length;
        int maxStackPos = lengthOffset + 4;
        int codeLengthPos = maxStackPos + 4; // max_stack, max_locals.
        int codeLength = readU4At(buf, codeLengthPos);
        if (codeLength + shift > 0xffff) {
            throw new ClassFormatError(
                    "entry counter would pass the 65535-byte code limit at " + codeLength);
        }
        int codeStart = codeLengthPos + 4;
        // The receiver shape pushes one ref and the value shape a ref and a
        // float, so a body provisioning less is raised to the shape's need.
        Edit maxStackEdit =
                minStack > 0 && readU2At(buf, maxStackPos) < minStack
                        ? new Edit(
                                maxStackPos,
                                maxStackPos + 2,
                                new byte[] {0x00, (byte) minStack})
                        : null;

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
                    int pc = ((fixed[at] & 0xff) << 8 | (fixed[at + 1] & 0xff)) + shift;
                    fixed[at] = (byte) ((pc >>> 8) & 0xff);
                    fixed[at + 1] = (byte) (pc & 0xff);
                }
            }
            exceptionEdit = new Edit(exceptionStart, exceptionStart + fixed.length, fixed);
        }

        int growth = 0;
        java.util.List<Edit> inner = new java.util.ArrayList<Edit>();
        int innerCountPos = exceptionStart + exceptionCount * 8;
        int innerCount = readU2At(buf, innerCountPos);
        int at = innerCountPos + 2;
        for (int a = 0; a < innerCount; a++) {
            int attrName = readU2At(buf, at);
            int attrLength = readU4At(buf, at + 2);
            if ("StackMapTable".equals(pool[attrName]) && readU2At(buf, at + 6) > 0) {
                int framePos = at + 8;
                byte[] head = shiftedFrameHead(buf, framePos, shift);
                int oldHead = (buf[framePos] & 0xff) <= 127 ? 1 : 3;
                growth = head.length - oldHead;
                if (growth != 0) {
                    inner.add(new Edit(at + 2, at + 6, u4(attrLength + growth)));
                }
                inner.add(new Edit(framePos, framePos + oldHead, head));
            }
            if ("LineNumberTable".equals(pool[attrName]) && readU2At(buf, at + 6) > 0) {
                inner.add(shiftedLineTable(buf, at + 8, readU2At(buf, at + 6), shift));
            }
            at += 6 + attrLength;
        }

        edits.add(new Edit(lengthOffset, lengthOffset + 4, u4(length + shift + growth)));
        if (maxStackEdit != null) {
            edits.add(maxStackEdit);
        }
        edits.add(new Edit(codeLengthPos, codeLengthPos + 4, u4(codeLength + shift)));
        edits.add(new Edit(codeStart, codeStart, entry));
        if (exceptionEdit != null) {
            edits.add(exceptionEdit);
        }
        edits.addAll(inner);
    }

    /**
     * Renders one LineNumberTable's entries with every {@code start_pc}
     * shifted, then re-anchors the lowest entry at zero so the injected
     * bytes stay covered by the first real line. Left unshifted, every
     * reported line in a patched method drifts to whichever entry the
     * shifted pc falls into next -- measured as a draw at real line 742
     * reported as 747, which sent a session tracing the wrong statement
     * (wiki log 2026-09-11).
     *
     * @param buf The class bytes.
     * @param tableStart Offset of the first entry's {@code start_pc}.
     * @param entries Number of entries; the caller has proven it non-zero.
     * @param shift The entry's length in bytes.
     * @return The replacement edit spanning exactly the entries.
     */
    private static Edit shiftedLineTable(byte[] buf, int tableStart, int entries, int shift) {
        byte[] table = new byte[entries * 4];
        System.arraycopy(buf, tableStart, table, 0, table.length);
        int lowestAt = 0;
        int lowestPc = Integer.MAX_VALUE;
        for (int e = 0; e < entries; e++) {
            int pcAt = e * 4;
            int pc = ((table[pcAt] & 0xff) << 8 | (table[pcAt + 1] & 0xff)) + shift;
            table[pcAt] = (byte) ((pc >>> 8) & 0xff);
            table[pcAt + 1] = (byte) (pc & 0xff);
            if (pc < lowestPc) {
                lowestPc = pc;
                lowestAt = pcAt;
            }
        }
        table[lowestAt] = 0;
        table[lowestAt + 1] = 0;
        return new Edit(tableStart, tableStart + table.length, table);
    }

    /**
     * Renders the first StackMapTable frame's head with its offset delta
     * grown by the shift -- the uniform displacement the prepend applies to
     * every bytecode offset. Only the FIRST frame carries an absolute
     * delta; every later frame is a delta from its predecessor and needs
     * nothing.
     *
     * @param buf The class bytes.
     * @param framePos Offset of the first frame's tag byte.
     * @param shift The entry's length in bytes.
     * @return The replacement head: same tag with the bigger compact delta,
     *     the delta bumped in place for the u2 forms, or the extended
     *     re-encoding when the compact form can no longer carry it.
     */
    private static byte[] shiftedFrameHead(byte[] buf, int framePos, int shift) {
        int tag = buf[framePos] & 0xff;
        if (tag <= 63) {
            int delta = tag + shift;
            if (delta <= 63) {
                return new byte[] {(byte) delta};
            }
            return new byte[] {(byte) 251, (byte) ((delta >>> 8) & 0xff), (byte) (delta & 0xff)};
        }
        if (tag <= 127) {
            int delta = tag - 64 + shift;
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
        int delta = readU2At(buf, framePos + 1) + shift;
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
