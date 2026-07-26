package rwbot.agent;

/**
 * Checks for {@link AgentOptions} parsing.
 *
 * <p>Weighted towards rejections. A misspelled option that silently does
 * nothing presents as "the agent ignored me" during a probe, which is the most
 * expensive kind of quiet failure to chase, so every malformed form is asserted
 * to be refused rather than merely not crash.
 */
final class OptionChecks {

    private OptionChecks() {
    }

    /** Exercises option parsing, including the rejections that must stay loud. */
    static int checkOptions() {
        int failures = 0;

        failures += Check.expect(!AgentOptions.parse(null).discoveryRequested(), "no argument -> no discovery");
        failures += Check.expect(!AgentOptions.parse("").discoveryRequested(), "blank argument -> no discovery");

        int[] parsed = AgentOptions.parse("discoverAtSeconds=20,5").discoverAtSeconds();
        failures += Check.expect(parsed.length == 2 && parsed[0] == 5 && parsed[1] == 20, "times parsed and sorted");

        failures += Check.expect(
                !AgentOptions.parse("discoverAtSeconds=5").exitAfterDiscovery(),
                "exitAfterDiscovery defaults off");
        failures += Check.expect(
                AgentOptions.parse("discoverAtSeconds=5;exitAfterDiscovery=true").exitAfterDiscovery(),
                "exitAfterDiscovery honoured");
        failures += expectRejected("exitAfterDiscovery=yes", "non-boolean exit flag");
        failures += Check.expect(
                AgentOptions.parse("inspectFields=X,W").inspectFields().length == 2,
                "inspectFields parsed");
        failures += expectRejected("inspectFields=X,,W", "blank field name");
        failures += Check.expect(
                "com.x".equals(AgentOptions.parse("findElementsUnder=com.x").findElementsUnder()),
                "findElementsUnder parsed");
        failures += Check.expect(
                AgentOptions.parse("discoverAtSeconds=5").findElementsUnder().isEmpty(),
                "findElementsUnder defaults empty");
        failures += Check.expect(
                !AgentOptions.parse("discoverAtSeconds=5").orderRequested(),
                "order not requested by default");
        failures += Check.expect(
                AgentOptions.parse("orderMoveAtSeconds=25").orderMoveAtSeconds() == 25,
                "orderMoveAtSeconds parsed");
        failures += Check.expect(
                AgentOptions.parse("orderMoveAtSeconds=25").orderMoveUnitIndex() == 0,
                "orderMoveUnitIndex defaults to 0");
        failures += Check.expect(
                AgentOptions.parse("orderMoveAtSeconds=25;orderMoveUnitIndex=2")
                                .orderMoveUnitIndex()
                        == 2,
                "orderMoveUnitIndex parsed");
        float[] moveBy = AgentOptions.parse("orderMoveBy=300,-40").orderMoveBy();
        failures += Check.expect(
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

    /** Asserts that an option string is rejected rather than silently accepted. */
    static int expectRejected(String argument, String description) {
        try {
            AgentOptions.parse(argument);
        } catch (IllegalArgumentException e) {
            return Check.expect(true, "rejects " + description);
        }
        return Check.expect(false, "rejects " + description);
    }
}
