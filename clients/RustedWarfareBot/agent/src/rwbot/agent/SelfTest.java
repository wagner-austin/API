package rwbot.agent;

import java.util.jar.JarFile;

/**
 * Verifies the patcher against the real pinned jar.
 *
 * <p>The oracle is the JVM's own bytecode verifier, not a second reading of the
 * bytes by the code that produced them: each patched class is <em>defined and
 * linked</em>, so a malformed constant pool, a bad attribute length or an
 * unbalanced stack fails here rather than at the first rendered frame.
 *
 * <p>Run via {@code make agent-selftest}. Exits non-zero on any failure.
 */
public final class SelfTest {

    private SelfTest() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("usage: SelfTest <path-to-game-lib.jar>");
            System.exit(2);
        }

        java.util.Map<String, java.util.Set<String>> targets = Targets.byClass();
        PatchingLoader loader = new PatchingLoader(SelfTest.class.getClassLoader());
        int failures = 0;

        JarFile jar = new JarFile(args[0]);
        try {
            for (java.util.Map.Entry<String, java.util.Set<String>> entry : targets.entrySet()) {
                String internalName = entry.getKey();
                java.util.Set<String> methods = entry.getValue();
                if (!check(jar, loader, internalName, methods)) {
                    failures++;
                }
            }
        } finally {
            jar.close();
        }

        if (failures > 0) {
            System.out.println("FAIL " + failures + " target(s)");
            System.exit(1);
        }
        System.out.println("OK " + targets.size() + " target(s) patched, defined and linked");
    }

    private static boolean check(
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

    private static byte[] readFully(JarFile jar, java.util.jar.JarEntry entry)
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
