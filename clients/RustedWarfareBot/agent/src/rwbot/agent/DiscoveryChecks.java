package rwbot.agent;

/**
 * Checks for {@link Discovery}, with the fixtures they need.
 *
 * <p>The fixtures are shaped after the engine rather than after convenience:
 * a subclass whose identifying state lives on its base, and a class extending a
 * JDK type as twelve engine classes extend {@code Thread}. Both encode a
 * failure that actually happened — a featureless-looking subclass sold a wrong
 * identification, and an unguarded hierarchy walk reflected into JDK internals.
 */
final class DiscoveryChecks {

    private DiscoveryChecks() {
    }

    /** Exercises the reflective snapshot against an object whose shape is known here. */
    static int checkDiscovery() {
        int failures = 0;
        failures += Check.expect(Snapshot.describe(null, "t=0s").contains("target is null"), "null target");

        String report = Snapshot.describe(new DiscoverySample(), "t=1s");
        failures += Check.expect(report.contains("=== discovery t=1s ==="), "snapshot is labelled");
        failures += Check.expect(report.contains("DiscoverySample"), "declaring class named");
        failures += Check.expect(report.contains("size=3"), "collection size reported");
        failures += Check.expect(report.contains("of=java.lang.String"), "element class reported");
        failures += Check.expect(report.contains("int[] len=2"), "array length reported");
        failures += Check.expect(report.contains("null"), "null field reported");
        failures += Check.expect(report.contains("\"hello\""), "string value reported");
        // By constant name, not by class. A state field that renders as its
        // enum class carries no information at all, which is what the first
        // AI zone dump ran into.
        failures += Check.expect(report.contains("Mood.CALM"), "enum value reported by name");
        failures += Check.expect(report.contains("static "), "static field marked");

        DiscoverySample sample = new DiscoverySample();
        failures += Check.expect(
                Snapshot.describeElements(null, "names").contains("target is null"),
                "element expansion handles a null target");
        failures += Check.expect(
                Snapshot.describeElements(sample, "nope").contains("no field named nope"),
                "element expansion names an absent field");
        failures += Check.expect(
                Snapshot.describeElements(sample, "text").contains("not a collection"),
                "element expansion rejects a non-collection");
        failures += Check.expect(
                Snapshot.describeElements(sample, "absent").contains("field is null"),
                "element expansion handles a null field");

        String elements = Snapshot.describeElements(sample, "names");
        failures += Check.expect(elements.contains("[0] java.lang.String"), "elements indexed and typed");
        failures += Check.expect(elements.contains("[2] "), "every element expanded");
        failures += Check.expect(
                elements.contains("[0] java.lang.String = \"a\""),
                "platform elements summarised, not reflected into");

        // A subclass whose identifying state lives on its base must not read as
        // featureless -- the failure mode that sold an earlier wrong finding.
        String inherited = Snapshot.describeElements(new SubclassHolder(), "items");
        failures += Check.expect(inherited.contains("ownField"), "element's own field listed");
        failures += Check.expect(inherited.contains("baseField"), "element's INHERITED field listed");

        // The hierarchy walk must stop at the platform boundary. Twelve engine
        // classes extend Thread, four extend Exception, and the master entity
        // list type extends AbstractList -- so a game object's JDK superclass is
        // the common case, not an edge one. Reflecting its internals warns on
        // JDK 13 and is denied later.
        String overJdk = Snapshot.describe(new ExtendsPlatform(), "t=2s");
        failures += Check.expect(overJdk.contains("gameField"), "own field listed over a JDK base");
        failures += Check.expect(!overJdk.contains("priority"), "JDK superclass field NOT reflected");
        failures += Check.expect(!overJdk.contains("eetop"), "JDK superclass internals NOT reflected");

        String overJdkGraph =
                GraphSearch.findCollections(new ExtendsPlatformHolder(), "java.lang.String", 4, 500);
        failures += Check.expect(
                !overJdkGraph.contains(".eetop") && !overJdkGraph.contains(".priority"),
                "graph search does not climb into a JDK superclass");

        failures += Check.expect(
                GraphSearch.findCollections(null, "com.x", 3, 100).contains("root is null"),
                "graph search handles a null root");
        String found = GraphSearch.findCollections(sample, "java.lang.String", 3, 500);
        failures += Check.expect(found.contains(".names"), "graph search finds a matching collection");
        failures += Check.expect(found.contains("visited "), "graph search reports its budget");
        failures += Check.expect(
                !GraphSearch.findCollections(sample, "no.such.pkg", 3, 500).contains(".names"),
                "graph search filters by element package");

        // A container must be traversed by its elements, never by its declared
        // fields. Reflecting into java.util internals warns on JDK 13 and is
        // denied in later releases, so these names appearing in a path would be
        // a regression to a route that stops working rather than a style point.
        String deep = GraphSearch.findCollections(new NestedSample(), "java.lang.String", 4, 500);
        failures += Check.expect(
                deep.contains(".holder[0].tag"),
                "graph search descends through a collection by element index");
        for (String internal : new String[] {"modCount", "elementData", "serialVersionUID"}) {
            failures += Check.expect(
                    !deep.contains(internal),
                    "graph search never walks the platform internal " + internal);
        }
        return failures;
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
        final Mood mood = Mood.CALM;
        final Object absent = null;
    }

    /** Stands in for the engine's obfuscated state enums. */
    private enum Mood {
        CALM
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
}
