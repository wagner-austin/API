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
        failures += checkOrderBindings();
        failures += checkStateStream();
        failures += checkCommandParsing();
        failures += checkChannelBackpressure();
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
        failures += expect(
                !AgentOptions.parse("discoverAtSeconds=5").orderRequested(),
                "order not requested by default");
        failures += expect(
                AgentOptions.parse("orderMoveAtSeconds=25").orderMoveAtSeconds() == 25,
                "orderMoveAtSeconds parsed");
        failures += expect(
                AgentOptions.parse("orderMoveAtSeconds=25").orderMoveUnitIndex() == 0,
                "orderMoveUnitIndex defaults to 0");
        failures += expect(
                AgentOptions.parse("orderMoveAtSeconds=25;orderMoveUnitIndex=2")
                                .orderMoveUnitIndex()
                        == 2,
                "orderMoveUnitIndex parsed");
        float[] moveBy = AgentOptions.parse("orderMoveBy=300,-40").orderMoveBy();
        failures += expect(
                moveBy[0] == 300.0f && moveBy[1] == -40.0f, "orderMoveBy parsed as x,y");
        failures += expectRejected("orderMoveAtSeconds=0", "zero order time");
        failures += expectRejected("orderMoveUnitIndex=-1", "negative roster index");
        failures += expectRejected("orderMoveBy=300", "one-component offset");
        failures += expectRejected("orderMoveBy=300,left", "non-numeric offset");
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

        // A subclass whose identifying state lives on its base must not read as
        // featureless -- the failure mode that sold an earlier wrong finding.
        String inherited = Discovery.describeElements(new SubclassHolder(), "items");
        failures += expect(inherited.contains("ownField"), "element's own field listed");
        failures += expect(inherited.contains("baseField"), "element's INHERITED field listed");

        // The hierarchy walk must stop at the platform boundary. Twelve engine
        // classes extend Thread, four extend Exception, and the master entity
        // list type extends AbstractList -- so a game object's JDK superclass is
        // the common case, not an edge one. Reflecting its internals warns on
        // JDK 13 and is denied later.
        String overJdk = Discovery.describe(new ExtendsPlatform(), "t=2s");
        failures += expect(overJdk.contains("gameField"), "own field listed over a JDK base");
        failures += expect(!overJdk.contains("priority"), "JDK superclass field NOT reflected");
        failures += expect(!overJdk.contains("eetop"), "JDK superclass internals NOT reflected");

        String overJdkGraph =
                Discovery.findCollections(new ExtendsPlatformHolder(), "java.lang.String", 4, 500);
        failures += expect(
                !overJdkGraph.contains(".eetop") && !overJdkGraph.contains(".priority"),
                "graph search does not climb into a JDK superclass");

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

    /**
     * Resolves every obfuscated name the order path uses, against the real jar.
     *
     * <p>No running game is needed: the classes, fields and method signatures
     * either exist in the pinned jar or they do not. This is the difference
     * between a game update failing at the gate with a list of what moved, and
     * failing mid-run with a reflection error nobody sees until they read a log.
     */
    private static int checkOrderBindings() {
        java.util.List<String> problems = Orders.verifyBindings();
        for (String problem : problems) {
            System.out.println("FAIL order binding: " + problem);
        }
        return expect(problems.isEmpty(), "every order-path name resolves against the jar");
    }

    /** Exercises the inbound order format, including every rejection. */
    private static int checkCommandParsing() {
        int failures = 0;

        CommandRecord move =
                CommandRecord.parse("{\"kind\":\"move\",\"unit_id\":214,\"x\":4550.0,\"y\":2610.5}");
        failures += expect(move.kind() == CommandRecord.Kind.MOVE, "move verb parsed");
        failures += expect(move.unitId() == 214L, "move unit id parsed");
        failures += expect(move.x() == 4550.0f && move.y() == 2610.5f, "move target parsed");
        failures += expect(move.buildType().isEmpty(), "a move carries no build type");

        CommandRecord build =
                CommandRecord.parse(
                        "{\"kind\":\"build\",\"unit_id\":215,\"x\":1.0,\"y\":2.0,"
                                + "\"type\":\"landFactory\"}");
        failures += expect(build.kind() == CommandRecord.Kind.BUILD, "build verb parsed");
        failures += expect("landFactory".equals(build.buildType()), "build type parsed");

        failures += expect(
                CommandRecord.parse("{\"kind\":\"move\",\"unit_id\":1,\"x\":-3,\"y\":4}").x()
                        == -3.0f,
                "an integer coordinate is accepted as a float");

        // A field that belongs to another verb is rejected rather than ignored:
        // silently dropping it would let a mistyped build read as a move.
        failures += expectBadCommand(
                "{\"kind\":\"move\",\"unit_id\":1,\"x\":1,\"y\":2,\"type\":\"landFactory\"}",
                "a move carrying a build type");
        failures += expectBadCommand("{\"kind\":\"fly\",\"unit_id\":1,\"x\":1,\"y\":2}", "unknown verb");
        failures += expectBadCommand("{\"kind\":\"move\",\"x\":1,\"y\":2}", "missing unit id");
        failures += expectBadCommand("{\"kind\":\"move\",\"unit_id\":1,\"y\":2}", "missing x");
        failures += expectBadCommand(
                "{\"kind\":\"build\",\"unit_id\":1,\"x\":1,\"y\":2}", "build with no type");
        failures += expectBadCommand(
                "{\"kind\":\"build\",\"unit_id\":1,\"x\":1,\"y\":2,\"type\":\"\"}",
                "build with a blank type");
        failures += expectBadCommand(
                "{\"kind\":\"move\",\"unit_id\":1,\"x\":NaN,\"y\":2}", "a non-finite coordinate");
        failures += expectBadCommand(
                "{\"kind\":\"move\",\"unit_id\":\"lots\",\"x\":1,\"y\":2}",
                "a non-numeric unit id");
        failures += expectBadCommand(
                "{\"kind\":\"move\",\"unit_id\":1,\"x\":1,\"y\":2}extra", "trailing text");
        failures += expectBadCommand(
                "{\"kind\":\"move\",\"unit_id\":1,\"x\":{\"v\":1},\"y\":2}", "a nested value");
        failures += expectBadCommand(
                "{\"kind\":\"move\",\"kind\":\"build\",\"unit_id\":1,\"x\":1,\"y\":2}",
                "a duplicate key");
        failures += expectBadCommand("not json at all", "text that is not an object");
        return failures;
    }

    /**
     * A slow planner must never stall the simulation.
     *
     * <p>The outbox drops its oldest sample when full rather than blocking the
     * game thread. Asserted rather than commented, because the failure mode --
     * a paused match whenever the planner is busy -- would be attributed to the
     * game long before it was attributed to the queue.
     */
    private static int checkChannelBackpressure() {
        int failures = 0;
        CommandChannel channel = new CommandChannel(0, 250);
        for (int i = 0; i < 4; i++) {
            failures += expect(channel.offer("sample " + i), "sample " + i + " queued without a drop");
        }
        failures += expect(!channel.offer("sample 4"), "the fifth sample reports a drop");
        failures += expect(channel.queued() == 4, "the outbox stays bounded at its depth");
        return failures;
    }

    /** Asserts one command line is rejected. */
    private static int expectBadCommand(String line, String what) {
        try {
            CommandRecord.parse(line);
        } catch (IllegalArgumentException e) {
            return expect(true, "rejects " + what);
        }
        return expect(false, "rejects " + what);
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

    /**
     * A game-shaped class over a JDK base, as twelve engine classes are over
     * {@code Thread}. Never started; only its fields are read.
     */
    private static final class ExtendsPlatform extends Thread {
        final String gameField = "mine";
    }

    /** Reaches an {@link ExtendsPlatform} through a field, for the graph search. */
    private static final class ExtendsPlatformHolder {
        final ExtendsPlatform child = new ExtendsPlatform();
    }

    /** Base carrying the state that identifies an element, as engine types do. */
    private static class Base {
        final int baseField = 7;
    }

    /** Subclass declaring little of its own, like {@code units.al} over {@code am}. */
    private static final class Derived extends Base {
        final String ownField = "own";
    }

    /** Holder whose collection elements are a subclass with an inherited field. */
    private static final class SubclassHolder {
        final java.util.List<Derived> items = java.util.Arrays.asList(new Derived());
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

    /**
     * Exercises the NDJSON writer. The consumer parses these lines strictly and
     * cannot fall back on a lenient JSON library, so malformed output here is a
     * broken contract rather than cosmetic.
     */
    private static int checkStateStream() {
        int failures = 0;

        String frame = StateStream.frameRecord(1918, 6461, 3);
        failures += expect(
                frame.equals("{\"kind\":\"frame\",\"frame\":1918,\"clock_ms\":6461,\"owned\":3}"),
                "frame record is exact");

        // The consumer splits on newlines before parsing, so a newline inside
        // a record would silently become two malformed ones. Code points
        // rather than character literals: 10 is LF, 13 is CR.
        failures += expect(
                frame.indexOf(10) < 0 && frame.indexOf(13) < 0,
                "a record never contains a newline");
        failures += expect(
                frame.startsWith("{") && frame.endsWith("}"),
                "a record is exactly one object");

        failures += expect(
                StateStream.frameRecord(0, 0, 0).contains("\"owned\":0"),
                "an empty roster is still a record");
        return failures;
    }
}
