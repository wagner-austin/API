package rwbot.agent;

/**
 * Replaces the engine's ONE inline {@code new java.util.Random()} with a
 * call to {@link SideDraw#aiChoice()} -- the wall-decoupling program's
 * second pin (wiki: policy-determinism, the 2026-09-11 seam-2 arc).
 *
 * <p>The AI's turret picker ({@code a.a.m}) draws its choice from a
 * generator it constructs inline, seeded by the JVM's default nanoTime
 * mix: a wall-seeded draw invisible to every tapped stream by
 * construction, measured as byte-identical twins entering the same tick
 * with bit-identical balances and choosing different turrets from the
 * same candidate list. The jar-wide census finds exactly one such site
 * in simulation code (map generation and audio carry the others), so the
 * pin is one seven-byte splice: {@code new Random(); dup; invokespecial
 * <init>} is exactly the length of {@code invokestatic aiChoice} plus
 * four {@code nop}s, both leave one reference on the stack, and no
 * branch can target the interior -- so nothing shifts, no frame moves,
 * and the verifier sees an ordinary static call.
 *
 * <p>Alters what the simulation consumes, so it takes the hosting
 * containment every sim-altering patch takes ({@link Premain}): a
 * private determinism fix desyncs against a stock-engine peer.
 */
final class RandomRedirect {

    private static final String RANDOM_CLASS = "java/util/Random";

    private RandomRedirect() {
    }

    /**
     * Returns a copy of {@code classFile} with the single
     * {@code new Random()} site redirected to the choice stream.
     *
     * @param classFile The class bytes as loaded.
     * @return The patched bytes, same length but for the appended pool.
     * @throws ClassFormatError if the class cannot be parsed, carries no
     *     such site, or carries more than one -- the census counted
     *     exactly one, and a moved build must fail at the gate rather
     *     than ship a half-pinned engine.
     */
    static byte[] redirect(byte[] classFile) {
        ClassFilePatcher patcher = new ClassFilePatcher(classFile);
        String[] pool = patcher.readHeaderAndConstantPool();
        int poolEnd = patcher.pos;
        int poolCount = patcher.tags.length;
        if (poolCount + 6 > 0xffff) {
            throw new ClassFormatError("constant pool too large to grow: " + poolCount);
        }

        PoolIndex index = new PoolIndex(classFile, patcher.tags, pool);
        int randomClass = index.classNamed(RANDOM_CLASS);
        int initRef = index.methodrefOf(randomClass, "<init>", "()V");
        if (randomClass < 0 || initRef < 0) {
            throw new ClassFormatError(
                    "no Random()/<init> pool entries -- the pinned engine build moved"
                            + " under the redirect");
        }

        // The 7-byte site: new #randomClass, dup, invokespecial #initRef.
        byte[] site = {
            (byte) 0xbb,
            (byte) ((randomClass >>> 8) & 0xff),
            (byte) (randomClass & 0xff),
            0x59,
            (byte) 0xb7,
            (byte) ((initRef >>> 8) & 0xff),
            (byte) (initRef & 0xff),
        };
        int at = -1;
        int count = 0;
        for (int i = poolEnd; i <= classFile.length - site.length; i++) {
            boolean hit = true;
            for (int j = 0; j < site.length; j++) {
                if (classFile[i + j] != site[j]) {
                    hit = false;
                    break;
                }
            }
            if (hit) {
                at = i;
                count++;
            }
        }
        if (count != 1) {
            throw new ClassFormatError(
                    "expected exactly one new-Random site, found " + count
                            + " -- the census counted one, so the pinned build moved");
        }

        // Appended pool: owner Utf8, owner Class, name Utf8, descriptor
        // Utf8, NameAndType, Methodref -- the aiChoice call target.
        java.io.ByteArrayOutputStream appended = new java.io.ByteArrayOutputStream();
        byte[] owner = "rwbot/agent/SideDraw".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        appended.write(ClassFilePatcher.CONSTANT_UTF8);
        CodeBodies.writeU2(appended, owner.length);
        appended.write(owner, 0, owner.length);
        appended.write(ClassFilePatcher.CONSTANT_CLASS);
        CodeBodies.writeU2(appended, poolCount);
        byte[] name = "aiChoice".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        appended.write(ClassFilePatcher.CONSTANT_UTF8);
        CodeBodies.writeU2(appended, name.length);
        appended.write(name, 0, name.length);
        byte[] descriptor =
                "()Ljava/util/Random;".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        appended.write(ClassFilePatcher.CONSTANT_UTF8);
        CodeBodies.writeU2(appended, descriptor.length);
        appended.write(descriptor, 0, descriptor.length);
        appended.write(ClassFilePatcher.CONSTANT_NAME_AND_TYPE);
        CodeBodies.writeU2(appended, poolCount + 2);
        CodeBodies.writeU2(appended, poolCount + 3);
        appended.write(ClassFilePatcher.CONSTANT_METHODREF);
        CodeBodies.writeU2(appended, poolCount + 1);
        CodeBodies.writeU2(appended, poolCount + 4);
        int callRef = poolCount + 5;

        int newCount = poolCount + 6;
        java.util.List<Edit> edits = new java.util.ArrayList<Edit>();
        edits.add(
                new Edit(
                        8,
                        10,
                        new byte[] {(byte) ((newCount >>> 8) & 0xff), (byte) (newCount & 0xff)}));
        edits.add(new Edit(poolEnd, poolEnd, appended.toByteArray()));
        edits.add(
                new Edit(
                        at,
                        at + site.length,
                        new byte[] {
                            (byte) 0xb8,
                            (byte) ((callRef >>> 8) & 0xff),
                            (byte) (callRef & 0xff),
                            0x00,
                            0x00,
                            0x00,
                            0x00,
                        }));
        return patcher.applyEdits(edits);
    }

    /**
     * A one-pass structured walk of the constant pool, recording what the
     * redirect must resolve: Utf8 strings, Class name links, NameAndType
     * pairs, and Methodrefs. {@link ClassFilePatcher} exposes the Utf8
     * strings alone, and the redirect needs the LINKED entries.
     */
    private static final class PoolIndex {

        private final int[] classNameIndex;
        private final int[][] methodref;
        private final int[][] nameAndType;
        private final String[] utf8;

        PoolIndex(byte[] buf, int[] tags, String[] utf8) {
            this.utf8 = utf8;
            int n = tags.length;
            classNameIndex = new int[n];
            methodref = new int[n][];
            nameAndType = new int[n][];
            int pos = 10;
            for (int i = 1; i < n; i++) {
                int tag = buf[pos] & 0xff;
                pos++;
                if (tag == ClassFilePatcher.CONSTANT_UTF8) {
                    pos += 2 + (((buf[pos] & 0xff) << 8) | (buf[pos + 1] & 0xff));
                } else if (tag == ClassFilePatcher.CONSTANT_CLASS) {
                    classNameIndex[i] = ((buf[pos] & 0xff) << 8) | (buf[pos + 1] & 0xff);
                    pos += 2;
                } else if (tag == ClassFilePatcher.CONSTANT_METHODREF
                        || tag == 9
                        || tag == 11) {
                    methodref[i] =
                            new int[] {
                                ((buf[pos] & 0xff) << 8) | (buf[pos + 1] & 0xff),
                                ((buf[pos + 2] & 0xff) << 8) | (buf[pos + 3] & 0xff),
                            };
                    pos += 4;
                } else if (tag == ClassFilePatcher.CONSTANT_NAME_AND_TYPE) {
                    nameAndType[i] =
                            new int[] {
                                ((buf[pos] & 0xff) << 8) | (buf[pos + 1] & 0xff),
                                ((buf[pos + 2] & 0xff) << 8) | (buf[pos + 3] & 0xff),
                            };
                    pos += 4;
                } else if (tag == 8 || tag == 16 || tag == 19 || tag == 20) {
                    pos += 2;
                } else if (tag == 15) {
                    pos += 3;
                } else if (tag == 3 || tag == 4 || tag == 17 || tag == 18) {
                    pos += 4;
                } else if (tag == 5 || tag == 6) {
                    pos += 8;
                    i++;
                } else {
                    throw new ClassFormatError("unknown constant pool tag " + tag);
                }
            }
        }

        int classNamed(String internalName) {
            for (int i = 1; i < classNameIndex.length; i++) {
                if (classNameIndex[i] != 0 && internalName.equals(utf8[classNameIndex[i]])) {
                    return i;
                }
            }
            return -1;
        }

        int methodrefOf(int classIndex, String name, String descriptor) {
            for (int i = 1; i < methodref.length; i++) {
                int[] ref = methodref[i];
                if (ref == null || ref[0] != classIndex) {
                    continue;
                }
                int[] nat = nameAndType[ref[1]];
                if (nat != null && name.equals(utf8[nat[0]]) && descriptor.equals(utf8[nat[1]])) {
                    return i;
                }
            }
            return -1;
        }
    }
}
