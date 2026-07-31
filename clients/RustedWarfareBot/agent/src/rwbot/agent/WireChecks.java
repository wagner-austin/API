package rwbot.agent;

/**
 * Checks for everything that crosses the wire: the outbound world stream, the
 * inbound command parser, channel backpressure, and log prefixing.
 *
 * <p>Grouped because they share one property — the far end parses this text
 * strictly and cannot fall back on a lenient library, so malformed output is a
 * broken contract rather than a cosmetic defect. The frame record is pinned
 * byte-for-byte on purpose: a field appearing, vanishing or being renamed must
 * fail here rather than at the far end of a socket.
 */
final class WireChecks {

    private WireChecks() {
    }

    /**
     * Exercises the NDJSON writer. The consumer parses these lines strictly and
     * cannot fall back on a lenient JSON library, so malformed output here is a
     * broken contract rather than cosmetic.
     */
    static int checkStateStream() {
        int failures = 0;

        String frame = StateStream.frameRecord(1918, 6461, 3, 2, 5, 6, 4000, false, false, 6);
        // Pinned byte-for-byte on purpose. This is a wire contract, and the
        // consumer parses it strictly, so a field appearing, vanishing or being
        // renamed must fail here rather than at the far end of a socket. That
        // is the opposite of pinning an observed value like a frame counter,
        // which changes every capture and carries no contract.
        failures += Check.expect(
                frame.equals(
                        "{\"kind\":\"frame\",\"frame\":1918,\"clock_ms\":6461,"
                                + "\"visible\":3,\"pools\":2,\"options\":5,\"players\":6,"
                                + "\"credits\":4000,"
                                + "\"defeated\":false,\"wiped\":false,\"players_left\":6}"),
                "frame record is exact");

        // The scoreboard the engine keeps and the bot used to regress for.
        // Pinned byte-for-byte like every other record the consumer counts.
        failures += Check.expect(
                StateStream.playerRecord(
                                1918,
                                2,
                                new Perception.PlayerStat(
                                        3, false, true, false, false, 12, 4200, 9100))
                        .equals(
                                "{\"kind\":\"player\",\"frame\":1918,\"index\":2,\"team\":3,"
                                        + "\"local\":false,\"hostile\":true,\"defeated\":false,"
                                        + "\"wiped\":false,\"income\":12,\"army_value\":4200,"
                                        + "\"building_value\":9100}"),
                "player record is exact");
        // A player who has just been eliminated is exactly who a report wants to
        // name, so their slot is carried rather than dropped.
        failures += Check.expect(
                StateStream.playerRecord(
                                1, 0, new Perception.PlayerStat(1, true, false, true, true, 0, 0, 0))
                        .contains("\"defeated\":true,\"wiped\":true"),
                "an eliminated player is still on the wire");

        // Pinned for the same reason, and separately: the consumer counts these
        // off the frame record, so the two must agree about what a sample
        // contains.
        String pool = StateStream.poolRecord(1918, 0, new MapTiles.Pool(115, 6, 2310.0f, 130.0f), 4);
        failures += Check.expect(
                pool.equals(
                        "{\"kind\":\"pool\",\"frame\":1918,\"index\":0,\"tile_x\":115,"
                                + "\"tile_y\":6,\"x\":2310.0,\"y\":130.0,\"group_land\":4}"),
                "pool record is exact");
        failures += Check.expect(
                pool.indexOf(10) < 0 && pool.indexOf(13) < 0,
                "a pool record never contains a newline");

        String option =
                StateStream.optionRecord(
                        1918, 0, new BuildOptions.Option(228L, "tank", "u_tank", false, true, true, 350));
        failures += Check.expect(
                option.equals(
                        "{\"kind\":\"option\",\"frame\":1918,\"index\":0,\"unit_id\":228,"
                                + "\"produces\":\"tank\",\"key\":\"u_tank\",\"placed\":false,"
                                + "\"available\":true,\"makes_something\":true,\"price\":350}"),
                "option record is exact");
        // Reported rather than filtered out. An action a unit has but cannot use
        // yet is a wait; one it does not have at all is a dead plan entry, and
        // the planner can only tell them apart if both are on the wire.
        failures += Check.expect(
                StateStream.optionRecord(
                                1, 0, new BuildOptions.Option(1L, "x", "u_x", true, false, true, 0))
                        .contains("\"available\":false"),
                "an option a unit cannot use yet says so rather than being omitted");
        // Which verb orders it. A structure is placed at a chosen point; a unit
        // is queued in the building that makes it, and the two are dispatched
        // differently, so the flag has to survive the wire.
        failures += Check.expect(
                StateStream.optionRecord(
                                1, 0, new BuildOptions.Option(1L, "x", "u_x", true, true, true, 0))
                        .contains("\"placed\":true"),
                "a placed option is marked as one");
        // An action that concerns no type at all reaches the wire rather than
        // being dropped. Filtering these in the agent is what hid upgrades: the
        // engine models them as conversions, and a planner cannot weigh an
        // action it was never told about (wiki: policy-holding-ground).
        failures += Check.expect(
                StateStream.optionRecord(
                                1, 0, new BuildOptions.Option(1L, "", "c_2", false, true, false, 2000))
                        .contains("\"produces\":\"\",\"key\":\"c_2\",\"placed\":false"),
                "an action concerning no type is published rather than dropped");
        failures += Check.expect(
                StateStream.optionRecord(
                                1, 0, new BuildOptions.Option(1L, "x", "u_x", false, true, false, 0))
                        .endsWith("\"makes_something\":false,\"price\":0}"),
                "whether an action makes something is the planner's to judge, not the agent's");
        // The engine's own charge for the action, on the wire. Price is the
        // only reading that tells a factory's tier upgrade from its rally
        // point -- both concern no type -- and the first live tech probe
        // spent four unlock budgets setting rally points for want of it
        // (wiki: mechanics-build-actions).
        failures += Check.expect(
                StateStream.optionRecord(
                                1, 0, new BuildOptions.Option(1L, "", "c_2", false, true, false, 2000))
                        .contains("\"price\":2000"),
                "an action's engine price survives the wire");

        // The placement flags are a separate stream with the same constraint.
        failures += Check.expect(
                TypeFlags.record(4, "extractorT1", true)
                        .equals(
                                "{\"kind\":\"unittype\",\"index\":4,\"name\":\"extractorT1\","
                                        + "\"needs_pool\":true}"),
                "unit-type record is exact");
        failures += Check.expect(
                TypeFlags.record(0, "landFactory", false).endsWith("\"needs_pool\":false}"),
                "a type that needs no pool says so rather than being omitted");
        failures += Check.expect(
                TypeFlags.record(0, "a\"b", false).contains("\"name\":\"a\\\"b\""),
                "a quote in a mod's type name is escaped");

        // Reach and the layers it reaches onto. The planner reads both off one
        // record because a range with no reachable layer describes a unit that
        // shoots nothing, and the two were read in one pass over one prototype.
        failures += Check.expect(
                TypeFlags.combatRecord(
                                7, "c_tank", new TypeFlags.Combat(130.0f, true, false, false, true))
                        .equals(
                                "{\"kind\":\"unitcombat\",\"index\":7,\"name\":\"c_tank\","
                                        + "\"attack_range\":130.0,\"hits_land\":true,"
                                        + "\"hits_air\":false,\"hits_underwater\":false,"
                                        + "\"hits_land_out_of_water\":true}"),
                "unit-combat record is exact");
        failures += Check.expect(
                TypeFlags.combatRecord(0, "builder", TypeFlags.Combat.UNARMED)
                        .equals(
                                "{\"kind\":\"unitcombat\",\"index\":0,\"name\":\"builder\","
                                        + "\"attack_range\":0.0,\"hits_land\":false,"
                                        + "\"hits_air\":false,\"hits_underwater\":false,"
                                        + "\"hits_land_out_of_water\":false}"),
                "an unarmed type reaches no layer rather than inheriting the base predicates");

        // The consumer splits on newlines before parsing, so a newline inside
        // a record would silently become two malformed ones. Code points
        // rather than character literals: 10 is LF, 13 is CR.
        failures += Check.expect(
                frame.indexOf(10) < 0 && frame.indexOf(13) < 0,
                "a record never contains a newline");
        failures += Check.expect(
                frame.startsWith("{") && frame.endsWith("}"),
                "a record is exactly one object");

        failures += Check.expect(
                StateStream.frameRecord(0, 0, 0, 0, 0, 0, 0, false, false, 0).contains("\"visible\":0"),
                "an empty roster is still a record");
        failures += Check.expect(
                StateStream.frameRecord(0, 0, 0, 0, 0, 0, 0, false, false, 0).contains("\"pools\":0"),
                "a map with no pool in sight is still a record");
        failures += Check.expect(
                StateStream.frameRecord(0, 0, 0, 0, 0, 0, 0, false, false, 0).contains("\"options\":0"),
                "a player who can make nothing is still a record");
        return failures;
    }

    /** Exercises the inbound order format, including every rejection. */
    static int checkCommandParsing() {
        int failures = 0;

        CommandRecord move =
                CommandRecord.parse("{\"kind\":\"move\",\"unit_id\":214,\"x\":4550.0,\"y\":2610.5}");
        failures += Check.expect(move.kind() == CommandRecord.Kind.MOVE, "move verb parsed");
        failures += Check.expect(move.unitId() == 214L, "move unit id parsed");
        failures += Check.expect(move.x() == 4550.0f && move.y() == 2610.5f, "move target parsed");
        failures += Check.expect(move.buildType().isEmpty(), "a move carries no build type");

        CommandRecord build =
                CommandRecord.parse(
                        "{\"kind\":\"build\",\"unit_id\":215,\"x\":1.0,\"y\":2.0,"
                                + "\"type\":\"landFactory\"}");
        failures += Check.expect(build.kind() == CommandRecord.Kind.BUILD, "build verb parsed");
        failures += Check.expect("landFactory".equals(build.buildType()), "build type parsed");

        CommandRecord produce =
                CommandRecord.parse(
                        "{\"kind\":\"produce\",\"unit_id\":228,\"type\":\"tank\"}");
        failures += Check.expect(
                produce.kind() == CommandRecord.Kind.PRODUCE, "produce verb parsed");
        failures += Check.expect("tank".equals(produce.buildType()), "produce type parsed");
        failures += Check.expect(produce.unitId() == 228L, "produce unit id parsed");
        // A position on a produce command would be a coordinate nothing reads,
        // so it is refused rather than ignored -- the same rule that keeps a
        // mistyped build from reading as a move.
        failures += expectBadCommand(
                "{\"kind\":\"produce\",\"unit_id\":1,\"type\":\"tank\",\"x\":1}",
                "a produce carrying a position");
        failures += expectBadCommand(
                "{\"kind\":\"produce\",\"unit_id\":1}", "produce with no type");
        failures += expectBadCommand(
                "{\"kind\":\"produce\",\"unit_id\":1,\"type\":\"\"}", "produce with a blank type");
        CommandRecord ability =
                CommandRecord.parse(
                        "{\"kind\":\"ability\",\"unit_id\":213,\"key\":\"c_2\"}");
        failures += Check.expect(
                ability.kind() == CommandRecord.Kind.ABILITY, "ability verb parsed");
        failures += Check.expect("c_2".equals(ability.actionKey()), "ability key parsed");
        failures += Check.expect(ability.unitId() == 213L, "ability unit id parsed");
        // The unit and the selector are the whole of the order; anything more
        // is a field nothing reads, refused rather than ignored.
        failures += expectBadCommand(
                "{\"kind\":\"ability\",\"unit_id\":1,\"key\":\"c_2\",\"x\":1}",
                "an ability carrying a position");
        failures += expectBadCommand(
                "{\"kind\":\"ability\",\"unit_id\":1,\"key\":\"c_2\",\"type\":\"tank\"}",
                "an ability carrying a type");
        failures += expectBadCommand(
                "{\"kind\":\"ability\",\"unit_id\":1}", "an ability with no key");
        failures += expectBadCommand(
                "{\"kind\":\"ability\",\"unit_id\":1,\"key\":\"\"}",
                "an ability with a blank key");
        failures += expectBadCommand(
                "{\"kind\":\"ability\",\"unit_id\":1,\"key\":\"c_2\",\"action\":1}",
                "an ability carrying the retired index field");

        failures += Check.expect(
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

    /** Asserts one command line is rejected. */
    static int expectBadCommand(String line, String what) {
        try {
            CommandRecord.parse(line);
        } catch (IllegalArgumentException e) {
            return Check.expect(true, "rejects " + what);
        }
        return Check.expect(false, "rejects " + what);
    }

    /**
     * A slow planner must never stall the simulation.
     *
     * <p>The outbox drops its oldest sample when full rather than blocking the
     * game thread. Asserted rather than commented, because the failure mode --
     * a paused match whenever the planner is busy -- would be attributed to the
     * game long before it was attributed to the queue.
     */
    static int checkChannelBackpressure() {
        int failures = 0;
        CommandChannel channel = new CommandChannel(0, 250, 0, false);
        for (int i = 0; i < 4; i++) {
            failures += Check.expect(channel.offer("sample " + i), "sample " + i + " queued without a drop");
        }
        failures += Check.expect(!channel.offer("sample 4"), "the fifth sample reports a drop");
        failures += Check.expect(channel.queued() == 4, "the outbox stays bounded at its depth");
        return failures;
    }

    /** The engine captures stdout, so every emitted line must carry the prefix. */
    static int checkLogPrefixing() {
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
        return Check.expect(
                lines.length == 3 && prefixed == 3,
                "every line of a multi-line message is prefixed");
    }
}
