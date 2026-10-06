import { describe, expect, it } from "vitest";

import { authFrame, captured, enterFrame, quitFrame, readLobbyFrame, selectFrame, type JoinConfirm, type LobbyHandler, type LobbyRoom } from "../src/lobby.js";

const text = (body: Uint8Array): string => new TextDecoder().decode(body);
const encode = (value: string): Uint8Array => new TextEncoder().encode(value);

/** A handler that records each reply it is handed, in order. */
class RecordingLobby implements LobbyHandler {
  public readonly replies: string[] = [];
  public readonly rooms: LobbyRoom[] = [];
  public readonly confirms: JoinConfirm[] = [];

  public room(room: LobbyRoom): void {
    this.rooms.push(room);
    this.replies.push("room");
  }

  public confirm(confirm: JoinConfirm): void {
    this.confirms.push(confirm);
    this.replies.push("confirm");
  }

  public entered(roomId: string): void {
    this.replies.push(`entered ${roomId}`);
  }

  public quit(): void {
    this.replies.push("quit");
  }
}

describe("the client's lobby frames", () => {
  it("are the page client's AUTH, select, enter and quit", () => {
    expect(text(authFrame("1001", "tok", "magic5"))).toBe("%AUTH !be 1001|tok|0 magic5");
    expect(text(selectFrame("5"))).toBe("*5");
    expect(text(enterFrame("5", 2))).toBe("+5|2|128|128|web");
    expect(text(quitFrame())).toBe("-");
  });

  it("refuse a field that would break the frame, and a troop that is not a team", () => {
    expect(() => authFrame("10|01", "tok", "m")).toThrow('LOBBY_FIELD: the account "10|01" is empty or holds a pipe or a space');
    expect(() => authFrame("1001", "", "m")).toThrow("LOBBY_FIELD: the token");
    expect(() => authFrame("1001", "tok", "m m")).toThrow("LOBBY_FIELD: the magic");
    expect(() => selectFrame("")).toThrow("LOBBY_FIELD: the room");
    expect(() => enterFrame("1", 4)).toThrow("LOBBY_TROOP: troop 4 is not a team (0 to 3)");
    expect(() => enterFrame("1", -1)).toThrow("LOBBY_TROOP");
    expect(() => enterFrame("1", 1.5)).toThrow("LOBBY_TROOP");
  });
});

describe("readLobbyFrame", () => {
  it("reads the server's room rows, join confirm, enter response and quit echo", () => {
    const lobby = new RecordingLobby();
    readLobbyFrame(encode("+1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026"), lobby);
    readLobbyFrame(encode("+5|World (field05)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026"), lobby);
    readLobbyFrame(encode("=5|Oct. 05, 2026|austin|3|9|9|9|9"), lobby);
    readLobbyFrame(encode("$5|0"), lobby);
    readLobbyFrame(encode("-"), lobby);
    expect(lobby.rooms).toEqual([
      { roomId: "1", name: "Practice", image: "field01.gif", practice: true },
      { roomId: "5", name: "World (field05)", image: "field05.gif", practice: false },
    ]);
    expect(lobby.confirms).toEqual([{ roomId: "5", name: "austin", rank: 3 }]);
    expect(lobby.replies).toEqual(["room", "room", "confirm", "entered 5", "quit"]);
  });

  it("refuses anything else by its text", () => {
    const lobby = new RecordingLobby();
    for (const frame of ["+1|Practice|1|modes|2|x|field01.gif|2026", "=5|date|austin|three|9|9|9|9", "$5|1", "A1"]) {
      expect(() => readLobbyFrame(encode(frame), lobby)).toThrow(`LOBBY_FRAME: ${JSON.stringify(frame)} is not a lobby reply`);
    }
    expect(lobby.replies).toEqual([]);
  });
});

describe("captured", () => {
  it("reads a group that took part and refuses one that did not", () => {
    const match = /^(a)(b)?$/.exec("a");
    expect(match).not.toBeNull();
    if (match !== null) {
      expect(captured(match, 1)).toBe("a");
      expect(() => captured(match, 2)).toThrow('LOBBY_FRAME: group 2 of "a" did not take part');
    }
  });
});
