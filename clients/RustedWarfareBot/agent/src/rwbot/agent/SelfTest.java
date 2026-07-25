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

        failures += checkOptions();
        failures += checkDiscovery();
        failures += checkLogPrefixing();

        if (failures > 0) {
            System.out.println("FAIL " + failures + " check(s)");
            System.exit(1);
        }
        System.out.println("OK " + targets.size() + " target(s) patched, defined and linked");
    }

    /** Exercises option parsing, including the rejections that must stay loud. */
    private static int checkOptions() {
        int failures = 0;

        failures += expect(!AgentOptions.parse(null).discoveryRequested(), "no argument -> no discovery");
        failures += expect(!AgentOptions.parse("").discoveryRequested(), "blank argument -> no discovery");

        int[] parsed = AgentOptions.parse("discoverAtSeconds=20,5").discoverAtSeconds();
        failures += expect(parsed.length == 2 && parsed[0] == 5 && parsed[1] == 20, "times parsed and sorted");

        failures += expect(
                !AgentOptions.parse("discoverAtSeconds=5").exitAfterDiscovery(),
                "exitAfterDiscovery defaults off");
        failures += expect(
                AgentOptions.parse("discoverAtSeconds=5;exitAfterDiscovery=true").exitAfterDiscovery(),
                "exitAfterDiscovery honoured");
        failures += expectRejected("exitAfterDiscovery=yes", "non-boolean exit flag");
        failures += expect(
                AgentOptions.parse("inspectFields=X,W").inspectFields().length == 2,
                "inspectFields parsed");
        failures += expectRejected("inspectFields=X,,W", "blank field name");
        failures += expect(
                "com.x".equals(AgentOptions.parse("findElementsUnder=com.x").findElementsUnder()),
                "findElementsUnder parsed");
        failures += expect(
                AgentOptions.parse("discoverAtSeconds=5").findElementsUnder().isEmpty(),
                "findElementsUnder defaults empty");
        failures += expectRejected("discoverAtSeconds=0", "zero seconds");
        failures += expectRejected("discoverAtSeconds=-3", "negative seconds");
        failures += expectRejected("discoverAtSeconds=soon", "non-numeric seconds");
        failures += expectRejected("unknownKey=1", "unknown key");
        failures += expectRejected("noEquals", "malformed pair");
        return failures;
    }

    /** Exercises the reflective snapshot against an object whose shape is known here. */
    private static int checkDiscovery() {
        int failures = 0;
        failures += expect(Discovery.describe(null, "t=0s").contains("target is null"), "null target");

        String report = Discovery.describe(new DiscoverySample(), "t=1s");
        failures += expect(report.contains("=== discovery t=1s ==="), "snapshot is labelled");
        failures += expect(report.contains("DiscoverySample"), "declaring class named");
        failures += expect(report.contains("size=3"), "collection size reported");
        failures += expect(report.contains("of=java.lang.String"), "element class reported");
        failures += expect(report.contains("int[] len=2"), "array length reported");
        failures += expect(report.contains("null"), "null field reported");
        failures += expect(report.contains("\"hello\""), "string value reported");
        failures += expect(report.contains("static "), "static field marked");

        DiscoverySample sample = new DiscoverySample();
        failures += expect(
                Discovery.describeElements(null, "names").contains("target is null"),
                "element expansion handles a null target");
        failures += expect(
                Discovery.describeElements(sample, "nope").contains("no field named nope"),
                "element expansion names an absent field");
        failures += expect(
                Discovery.describeElements(sample, "text").contains("not a collection"),
                "element expansion rejects a non-collection");
        failures += expect(
                Discovery.describeElements(sample, "absent").contains("field is null"),
                "element expansion handles a null field");

        String elements = Discovery.describeElements(sample, "names");
        failures += expect(elements.contains("[0] java.lang.String"), "elements indexed and typed");
        failures += expect(elements.contains("[2] "), "every element expanded");
        failures += expect(
                elements.contains("[0] java.lang.String = \"a\""),
                "platform elements summarised, not reflected into");

        failures += expect(
                Discovery.findCollections(null, "com.x", 3, 100).contains("root is null"),
                "graph search handles a null root");
        String found = Discovery.findCollections(sample, "java.lang.String", 3, 500);
        failures += expect(found.contains(".names"), "graph search finds a matching collection");
        failures += expect(found.contains("visited "), "graph search reports its budget");
        failures += expect(
                !Discovery.findCollections(sample, "no.such.pkg", 3, 500).contains(".names"),
                "graph search filters by element package");

        // A container must be traversed by its elements, never by its declared
        // fields. Reflecting into java.util internals warns on JDK 13 and is
        // denied in later releases, so these names appearing in a path would be
        // a regression to a route that stops working rather than a style point.
        String deep = Discovery.findCollections(new NestedSample(), "java.lang.String", 4, 500);
        failures += expect(
                deep.contains(".holder[0].tag"),
                "graph search descends through a collection by element index");
        for (String internal : new String[] {"modCount", "elementData", "serialVersionUID"}) {
            failures += expect(
                    !deep.contains(internal),
                    "graph search never walks the platform internal " + internal);
        }
        return failures;
    }

    /** The engine captures stdout, so every emitted line must carry the prefix. */
    private static int checkLogPrefixing() {
        java.io.PrintStream original = System.out;
        java.io.ByteArrayOutputStream captured = new java.io.ByteArrayOutputStream();
        System.setOut(new java.io.PrintStream(captured, true));
        try {
            Log.info("first\nsecond\nthird");
        } finally {
            System.setOut(original);
        }
        String[] lines = captured.toString().split("\\R");
        int prefixed = 0;
        for (String line : lines) {
            if (line.startsWith("[rw-agent] ")) {
                prefixed++;
            }
        }
        return expect(
                lines.length == 3 && prefixed == 3,
                "every line of a multi-line message is prefixed");
    }

    /** Reports one assertion, returning 1 when it failed. */
    private static int expect(boolean condition, String description) {
        if (condition) {
            System.out.println("ok   " + description);
            return 0;
        }
        System.out.println("FAIL " + description);
        return 1;
    }

    /** Asserts that an option string is rejected rather than silently accepted. */
    private static int expectRejected(String argument, String description) {
        try {
            AgentOptions.parse(argument);
        } catch (IllegalArgumentException e) {
            return expect(true, "rejects " + description);
        }
        return expect(false, "rejects " + description);
    }

    /** Fixture with one field of every shape {@link Discovery} summarises. */
    private static final class DiscoverySample {
        static final String MARKER = "marker";
        final java.util.List<String> names = java.util.Arrays.asList("a", "b", "c");
        final int[] counts = {1, 2};
        final String text = "hello";
        final Object absent = null;
    }

    /** A non-platform object reachable only by descending through a collection. */
    private static final class Leaf {
        final java.util.List<String> tag = java.util.Arrays.asList("deep");
    }

    /**
     * Mirrors the engine shape the search exists for: a collection of game
     * objects, each owning the collection actually being hunted.
     */
    private static final class NestedSample {
        final java.util.List<Leaf> holder = java.util.Arrays.asList(new Leaf());
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
