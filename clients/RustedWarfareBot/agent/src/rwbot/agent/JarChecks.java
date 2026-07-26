package rwbot.agent;

import java.util.jar.JarFile;

/**
 * The checks that require the real pinned game jar.
 *
 * <p>Grouped by what they need rather than by what they touch. Everything here
 * fails for the same reason — a game update moved something — and everything
 * here is useless without {@code .game/game-lib.jar} on the classpath. The
 * other check groups are pure and run against fixtures.
 *
 * <p>Two halves. The patcher check defines and links each patched class, so
 * HotSpot's own bytecode verifier passes judgement rather than a second reading
 * by the code that wrote the bytes. The binding check resolves every pinned
 * obfuscated name against the jar with no game running, so a rename fails at
 * {@code make check} instead of in a live run.
 */
final class JarChecks {

    private JarChecks() {
    }

    /**
     * Patches every target in the real jar and verifies each result loads.
     *
     * @param jarPath Path to the pinned {@code game-lib.jar}.
     * @param targets Class-to-methods map to patch.
     * @return The number of targets that failed.
     * @throws java.io.IOException When the jar cannot be read.
     */
    static int checkPatcher(String jarPath, java.util.Map<String, java.util.Set<String>> targets)
            throws java.io.IOException {
        PatchingLoader loader = new PatchingLoader(JarChecks.class.getClassLoader());
        int failures = 0;
        JarFile jar = new JarFile(jarPath);
        try {
            for (java.util.Map.Entry<String, java.util.Set<String>> entry : targets.entrySet()) {
                if (!check(jar, loader, entry.getKey(), entry.getValue())) {
                    failures++;
                }
            }
        } finally {
            jar.close();
        }
        return failures;
    }

    static boolean check(
            JarFile jar,
            PatchingLoader loader,
            String internalName,
            java.util.Set<String> methods)
            throws java.io.IOException {

        java.util.jar.JarEntry entry = jar.getJarEntry(internalName + ".class");
        if (entry == null) {
            System.out.println("FAIL " + internalName + ": not present in jar");
            return false;
        }

        byte[] original = readFully(jar, entry);
        byte[] patched;
        try {
            patched = ClassFilePatcher.noOpMethods(original, methods);
        } catch (ClassFormatError e) {
            System.out.println("FAIL " + internalName + ": parse error: " + e.getMessage());
            return false;
        }

        if (patched == null) {
            System.out.println("FAIL " + internalName + ": no method matched " + methods);
            return false;
        }

        try {
            // defineClass + resolveClass forces linking, which is where HotSpot
            // runs the bytecode verifier over every method body in the class.
            loader.definePatched(internalName.replace('/', '.'), patched);
        } catch (LinkageError e) {
            System.out.println("FAIL " + internalName + ": did not verify: " + e);
            return false;
        }

        System.out.println(
                "ok   " + internalName + " " + methods
                        + "  (" + original.length + " -> " + patched.length + " bytes)");
        return true;
    }

    static byte[] readFully(JarFile jar, java.util.jar.JarEntry entry)
            throws java.io.IOException {
        java.io.InputStream in = jar.getInputStream(entry);
        try {
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int read;
            while ((read = in.read(chunk)) != -1) {
                out.write(chunk, 0, read);
            }
            return out.toByteArray();
        } finally {
            in.close();
        }
    }

    /**
     * Resolves every obfuscated name the order path uses, against the real jar.
     *
     * <p>No running game is needed: the classes, fields and method signatures
     * either exist in the pinned jar or they do not. This is the difference
     * between a game update failing at the gate with a list of what moved, and
     * failing mid-run with a reflection error nobody sees until they read a log.
     */
    static int checkOrderBindings() {
        java.util.List<String> problems = BindingCheck.verifyBindings();
        for (String problem : problems) {
            System.out.println("FAIL order binding: " + problem);
        }
        return Check.expect(problems.isEmpty(), "every order-path name resolves against the jar");
    }

    /** Defines patched classes in a child loader so the originals stay untouched. */
    private static final class PatchingLoader extends ClassLoader {

        PatchingLoader(ClassLoader parent) {
            super(parent);
        }

        void definePatched(String binaryName, byte[] bytes) {
            Class<?> defined = defineClass(binaryName, bytes, 0, bytes.length);
            resolveClass(defined);
        }
    }
}
