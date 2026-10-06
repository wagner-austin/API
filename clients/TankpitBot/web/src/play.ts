/**
 * The play page: join a room of the sim server and play it in the renderer.
 *
 * play.html is served by tankpit-sim-serve itself (the sim's net_web.py),
 * so everything the page needs is relative to its own URL: the static key
 * at cipher-key, the room's terrain at terrain/<room>, and the game's
 * socket at the page's own directory with a ws: or wss: scheme. Behind
 * Traefik that directory is /tankpit-sim/; served bare it is /.
 *
 * The form names the account and its token (issued by
 * tankpit-sim-accounts, never anyone's game account), the room and the
 * troop. Joining fetches the key and the terrain, builds the default pack,
 * and opens the connection; a click on the field moves the tank to that
 * tile, a click on the strip presses its button, and #status says where
 * the connection is after every event.
 */

import { hooks } from "./_test_hooks.js";
import { FieldTerrain } from "./field.js";
import { GAME_HEIGHT } from "./layers.js";
import { buildDefaultPack } from "./preview.js";
import { Renderer } from "./renderer.js";
import { RenderError } from "./scaled_context.js";
import { Phase, WireClient, type ClientListener, type JoinOptions } from "./wire_client.js";
import { WorldView } from "./world_view.js";

/** The part of a window the play page reads; the real window is one. */
export interface PlayPage {
  readonly document: Document;
  readonly location: { readonly href: string };
  readonly devicePixelRatio: number;
}

/** Who joins which room, on which troop. */
export interface JoinRequest {
  readonly account: string;
  readonly token: string;
  readonly roomId: string;
  readonly troop: number;
}

/** A game in play: its connection, view and renderer. */
export interface PlaySession {
  readonly client: WireClient;
  readonly view: WorldView;
  readonly renderer: Renderer;
}

/** Where a page served at a directory reads from. */
export interface ServerUrls {
  readonly key: string;
  readonly terrain: string;
  readonly socket: string;
}

const DECODER = new TextDecoder("utf-8");

/**
 * The URLs a page served at a directory reads from.
 *
 * @param href - The page's own URL.
 * @param roomId - The room it joins.
 * @returns The key's, the terrain's and the socket's URLs.
 */
export function serverUrls(href: string, roomId: string): ServerUrls {
  const base = new URL("./", href);
  const socket = new URL(base.href);
  socket.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return {
    key: new URL("cipher-key", base).href,
    terrain: new URL(`terrain/${encodeURIComponent(roomId)}`, base).href,
    socket: socket.href,
  };
}

/**
 * Write where a connection is into a status element.
 *
 * @param status - The element.
 * @returns The listener.
 */
export function statusListener(status: HTMLElement): ClientListener {
  return {
    changed(client: WireClient): void {
      const frame = client.frame;
      const drawn = frame === null ? "" : `, last frame ${frame.tilesDrawn} tiles and ${frame.tanksDrawn} tanks`;
      const closed = client.phase === Phase.Closed ? ` (${client.closedWith})` : "";
      status.textContent = `${client.phase}${closed}${drawn}`;
    },
  };
}

/**
 * Join a room and play it into an element.
 *
 * @param page - The page's window.
 * @param game - The element the game is drawn in.
 * @param status - The element that says where the connection is.
 * @param request - Who joins which room.
 * @returns The game in play; its socket is opening.
 * @throws Error FETCH_STATUS when the key or terrain does not answer 2xx; WireError for a terrain that does not decode.
 */
export async function joinGame(page: PlayPage, game: HTMLElement, status: HTMLElement, request: JoinRequest): Promise<PlaySession> {
  const urls = serverUrls(page.location.href, request.roomId);
  const staticKey = DECODER.decode(await hooks().fetchBytes(urls.key)).trim();
  const field = FieldTerrain.decode(await hooks().fetchBytes(urls.terrain));
  const manifest = buildDefaultPack(page.document);
  const renderer = await Renderer.create(game, page.devicePixelRatio, manifest);
  const view = new WorldView(field, renderer);
  const options: JoinOptions = { ...request, magic: hooks().randomMagic(), staticKey };
  const client = new WireClient((handlers) => hooks().openSocket(urls.socket, handlers), options, view, statusListener(status));
  game.addEventListener("click", (event) => {
    const bounds = game.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    if (y >= GAME_HEIGHT) {
      renderer.clickToolbar(x, y - GAME_HEIGHT);
      return;
    }
    client.moveTo(Math.floor(x / manifest.tileWidth), Math.floor(y / manifest.tileHeight));
  });
  return { client, view, renderer };
}

function element<T extends HTMLElement>(document: Document, id: string, kind: new () => T): T {
  const found = document.getElementById(id);
  if (!(found instanceof kind)) {
    throw new RenderError(`RENDER_PAGE: the page has no #${id} ${kind.name}`);
  }
  return found;
}

/**
 * Start the play page: read the join form on submit and join.
 *
 * @param page - The page's window.
 * @returns A promise of the session the first submit starts.
 * @throws RenderError RENDER_PAGE when the page lacks #game, #status or the #join form and its fields.
 */
export function startPlay(page: PlayPage): Promise<PlaySession> {
  const document = page.document;
  const game = element(document, "game", HTMLElement);
  const status = element(document, "status", HTMLElement);
  const form = element(document, "join", HTMLFormElement);
  const account = element(document, "account", HTMLInputElement);
  const token = element(document, "token", HTMLInputElement);
  const room = element(document, "room", HTMLInputElement);
  const troop = element(document, "troop", HTMLInputElement);
  return new Promise((resolve) => {
    form.addEventListener(
      "submit",
      (event) => {
        event.preventDefault();
        form.hidden = true;
        const request = { account: account.value.trim(), token: token.value.trim(), roomId: room.value.trim(), troop: Number(troop.value) };
        resolve(joinGame(page, game, status, request));
      },
      { once: true },
    );
  });
}
