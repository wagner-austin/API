package rwbot.agent;

/**
 * Clears {@code ACC_FINAL} on named static fields of a loaded class -- the
 * wall-decoupling program's pin (wiki: policy-determinism, the 2026-09-11
 * scan-pair entry).
 *
 * <p>The engine's generator holder declares its {@code java.util.Random}
 * {@code static final}, and HotSpot treats a static final reference of an
 * initialized class as a compile-time constant: a draw helper JIT-compiled
 * BEFORE the match-start generator swap keeps the ORIGINAL object baked
 * into its compiled body, and the {@code Unsafe} field write that installs
 * the seeded replacement deoptimizes nothing. Whether each helper crossed
 * its compilation threshold before the swap depends on how long and how
 * fast the menu demo ran -- wall clock -- so one byte-identical twin draws
 * from the seeded replacement while the other draws, invisibly, from a
 * generator whose state still carries the menu's consumption. That is a
 * real simulation fork (measured: the tsa/tsb pair, identical through
 * t=102, credits apart at t=103 with math and shuffle streams still
 * identical), and it is also every "one twin drew at the gate and the
 * other did not" tap asymmetry the fork hunt chased.
 *
 * <p>Clearing the final bit at class load closes the whole class of
 * failure: a non-final static is never constant-folded, so every read is a
 * real {@code getstatic} and the swap reaches all callers, compiled or
 * not. The field's runtime semantics are otherwise identical -- same
 * writes, same reads, same values -- so the patch is simulation-neutral
 * until the swap itself, which is the already-audited seeding path
 * (multiplayer containment unchanged; see {@link MatchSetup}).
 */
final class Definal {

    private static final int ACC_FINAL = 0x0010;

    private Definal() {
    }

    /**
     * Returns a copy of {@code classFile} with {@code ACC_FINAL} cleared on
     * every field named in {@code fields}.
     *
     * @param classFile The class bytes as loaded.
     * @param fields Field names to their required descriptors -- the
     *     descriptor check is what keeps a one-letter obfuscated name from
     *     silently matching a different field in a moved build.
     * @return The patched bytes, identical but for the cleared flag bits.
     * @throws ClassFormatError if the class cannot be parsed, a named field
     *     is absent, carries a different descriptor, or is not final --
     *     each means the pinned build moved, and a loud stop at class load
     *     beats a pin that silently pins nothing.
     */
    static byte[] strip(byte[] classFile, java.util.LinkedHashMap<String, String> fields) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        String[] pool = patcher.readHeaderAndConstantPool();
        patcher.skip(2); // access_flags
        patcher.skip(2); // this_class
        patcher.skip(2); // super_class
        int interfaceCount = patcher.readU2();
        patcher.skip(interfaceCount * 2);

        java.util.Set<String> unmatched = new java.util.LinkedHashSet<String>(fields.keySet());
        java.util.List<Edit> edits = new java.util.ArrayList<Edit>();
        int fieldCount = patcher.readU2();
        for (int i = 0; i < fieldCount; i++) {
            int flagsPos = patcher.pos;
            int flags = patcher.readU2();
            String name = pool[patcher.readU2()];
            String descriptor = pool[patcher.readU2()];
            int attributeCount = patcher.readU2();
            for (int a = 0; a < attributeCount; a++) {
                patcher.skip(2);
                patcher.skip(patcher.readU4());
            }
            String required = fields.get(name);
            if (required == null) {
                continue;
            }
            if (!required.equals(descriptor)) {
                throw new ClassFormatError(
                        "field " + name + " holds " + descriptor + ", not " + required
                                + " -- the pinned engine build moved under the pin");
            }
            if ((flags & ACC_FINAL) == 0) {
                throw new ClassFormatError(
                        "field " + name + " is already non-final"
                                + " -- the pinned engine build moved under the pin");
            }
            int cleared = flags & ~ACC_FINAL;
            edits.add(
                    new Edit(
                            flagsPos,
                            flagsPos + 2,
                            new byte[] {(byte) ((cleared >>> 8) & 0xff), (byte) (cleared & 0xff)}));
            unmatched.remove(name);
        }
        if (!unmatched.isEmpty()) {
            throw new ClassFormatError(
                    "no field matched definal target(s) " + unmatched
                            + " -- the pinned engine build moved under the pin");
        }
        return patcher.applyEdits(edits);
    }
}
