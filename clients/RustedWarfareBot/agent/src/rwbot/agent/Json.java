package rwbot.agent;

/**
 * A strict reader for one flat JSON object.
 *
 * <p>Not a JSON library. It accepts exactly the shape both directions of the
 * wire are constrained to -- an object whose keys are strings and whose values
 * are strings, numbers, or the literals {@code true}, {@code false} and
 * {@code null} -- and rejects everything else. Nesting and arrays are errors,
 * not unsupported features.
 *
 * <p>Written rather than vendored because the agent loads into the game's
 * classloader beside obfuscated classes, where every dependency is a conflict
 * surface (wiki: runtime-split-java-agent-python-brain). The subset is small
 * enough that a hand-written reader is auditable; a general one would not be.
 *
 * <p>Values are returned as their source text. A caller that wants a number
 * parses it and decides what a bad one means, so this reader never has to
 * guess whether {@code 1e999} is a number or a mistake.
 */
final class Json {

    private Json() {
    }

    /**
     * Reads one flat JSON object into key-to-source-text pairs.
     *
     * @param text A single JSON object. Surrounding whitespace is allowed.
     * @return Field names mapped to their unparsed value text, in encounter
     *     order. A JSON {@code null} maps to the text {@code "null"}.
     * @throws IllegalArgumentException When the text is not exactly one flat
     *     object of scalar values, or a key repeats.
     */
    static java.util.Map<String, String> flatObject(String text) {
        if (text == null) {
            throw new IllegalArgumentException("expected a JSON object, got nothing");
        }
        Cursor cursor = new Cursor(text);
        cursor.skipWhitespace();
        cursor.expect('{');
        java.util.Map<String, String> fields = new java.util.LinkedHashMap<String, String>();

        cursor.skipWhitespace();
        if (cursor.peek() == '}') {
            cursor.next();
            cursor.expectEnd();
            return fields;
        }

        while (true) {
            cursor.skipWhitespace();
            String key = cursor.readString();
            cursor.skipWhitespace();
            cursor.expect(':');
            cursor.skipWhitespace();
            String value = cursor.readScalar();
            if (fields.put(key, value) != null) {
                throw new IllegalArgumentException(
                        "duplicate key '" + key + "' in: " + text);
            }
            cursor.skipWhitespace();
            char separator = cursor.next();
            if (separator == '}') {
                cursor.expectEnd();
                return fields;
            }
            if (separator != ',') {
                throw new IllegalArgumentException(
                        "expected ',' or '}' at offset " + cursor.offset() + " in: " + text);
            }
        }
    }

    /**
     * Writes a JSON string literal, escaping what the grammar requires.
     *
     * <p>Every producer in the agent writes through this one. It was written
     * three times before it was written once -- {@link StateStream} and
     * {@link TypeFlags} each carried a private copy, and they had already
     * diverged on control characters -- which is the ordinary way a wire format
     * acquires two dialects. The consumer is a strict parser with no tolerance
     * for either.
     *
     * @param out Buffer to append to.
     * @param text The raw string.
     */
    static void quote(StringBuilder out, String text) {
        out.append('"');
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                    break;
            }
        }
        out.append('"');
    }

    /** A position in the source text, with the reads the grammar needs. */
    private static final class Cursor {

        private final String text;
        private int at;

        Cursor(String text) {
            this.text = text;
            this.at = 0;
        }

        int offset() {
            return at;
        }

        void skipWhitespace() {
            while (at < text.length()) {
                char c = text.charAt(at);
                if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                    return;
                }
                at++;
            }
        }

        char peek() {
            if (at >= text.length()) {
                throw new IllegalArgumentException("unexpected end of input in: " + text);
            }
            return text.charAt(at);
        }

        char next() {
            char c = peek();
            at++;
            return c;
        }

        void expect(char expected) {
            char actual = next();
            if (actual != expected) {
                throw new IllegalArgumentException(
                        "expected '" + expected + "' at offset " + (at - 1) + " in: " + text);
            }
        }

        void expectEnd() {
            skipWhitespace();
            if (at != text.length()) {
                throw new IllegalArgumentException(
                        "trailing text after the object at offset " + at + " in: " + text);
            }
        }

        String readString() {
            expect('"');
            StringBuilder out = new StringBuilder();
            while (true) {
                char c = next();
                if (c == '"') {
                    return out.toString();
                }
                if (c != '\\') {
                    out.append(c);
                    continue;
                }
                char escape = next();
                switch (escape) {
                    case '"':
                        out.append('"');
                        break;
                    case '\\':
                        out.append('\\');
                        break;
                    case '/':
                        out.append('/');
                        break;
                    case 'n':
                        out.append('\n');
                        break;
                    case 'r':
                        out.append('\r');
                        break;
                    case 't':
                        out.append('\t');
                        break;
                    default:
                        throw new IllegalArgumentException(
                                "unsupported escape '\\"
                                        + escape
                                        + "' at offset "
                                        + (at - 1)
                                        + " in: "
                                        + text);
                }
            }
        }

        /** Reads a string, number or literal. Rejects a nested object or array. */
        String readScalar() {
            char c = peek();
            if (c == '"') {
                return readString();
            }
            if (c == '{' || c == '[') {
                throw new IllegalArgumentException(
                        "nested values are not part of the wire format, at offset "
                                + at
                                + " in: "
                                + text);
            }
            int start = at;
            while (at < text.length()) {
                char here = text.charAt(at);
                if (here == ',' || here == '}' || here == ' ' || here == '\t'
                        || here == '\r' || here == '\n') {
                    break;
                }
                at++;
            }
            if (at == start) {
                throw new IllegalArgumentException(
                        "expected a value at offset " + at + " in: " + text);
            }
            return text.substring(start, at);
        }
    }
}
