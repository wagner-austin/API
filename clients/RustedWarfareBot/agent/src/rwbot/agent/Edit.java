package rwbot.agent;

/**
 * A byte range in an original class file and the bytes replacing it.
 *
 * <p>Was {@link ClassFilePatcher}'s inner class until the entry-count patch
 * ({@link EntryCounts}) crossed the module ceiling and moved out with it:
 * both patch families collect these against ORIGINAL offsets and let
 * {@link ClassFilePatcher#applyEdits} splice last-to-first, so no edit can
 * shift the offsets recorded for another.
 */
final class Edit {
    final int start;
    final int end;
    final byte[] replacement;

    Edit(int start, int end, byte[] replacement) {
        this.start = start;
        this.end = end;
        this.replacement = replacement;
    }
}
