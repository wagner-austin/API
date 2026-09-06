package rwbot.agent;

/**
 * The agent's self-check gate.
 *
 * <p>Orchestration only: it runs each group of checks, sums the failures and
 * sets the exit code. Every assertion lives with the concern it verifies —
 * {@link JarChecks} for what needs the real pinned jar, {@link OptionChecks},
 * {@link DiscoveryChecks} and {@link WireChecks} for the rest — so a group can
 * grow without this file growing with it.
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
        int failures = JarChecks.checkPatcher(args[0], targets);

        // The wall-paced spawners, held to the real jar for the same reason
        // and one more: their descriptors were derived by hand from javap
        // output, so a mistyped one would fail at premain on a compute node
        // rather than here. Checked against the pinned jar, a wrong name
        // fails the build.
        failures += JarChecks.checkPatcher(
                args[0],
                java.util.Collections.singletonMap(
                        Targets.EFFECT_SPAWNER_CLASS, Targets.effectSpawners()));

        failures += JarChecks.checkSyncPath(args[0]);
        failures += JarChecks.checkSwayRewire(args[0]);
        failures += OptionChecks.checkOptions();
        failures += DiscoveryChecks.checkDiscovery();
        failures += JarChecks.checkOrderBindings();
        failures += WireChecks.checkStateStream();
        failures += WireChecks.checkCommandParsing();
        failures += WireChecks.checkChannelBackpressure();
        failures += WireChecks.checkLogPrefixing();
        failures += RandomChecks.checkSplitRouting();

        if (failures > 0) {
            System.out.println("FAIL " + failures + " check(s)");
            System.exit(1);
        }
        System.out.println("OK " + targets.size() + " target(s) patched, defined and linked");
    }
}
