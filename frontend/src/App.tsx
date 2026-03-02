import { useCallback, useState } from "react";
import GameView from "./components/GameView";
import Lobby from "./components/Lobby";
import type { JoinResponse } from "./types/game";

interface ActiveGame {
  joinInfo: JoinResponse;
  isHost: boolean;
}

export default function App() {
  const [activeGame, setActiveGame] = useState<ActiveGame | null>(
    null,
  );

  const handleJoined = useCallback(
    (joinInfo: JoinResponse, isHost: boolean) => {
      setActiveGame({ joinInfo, isHost });
    },
    [],
  );

  const handleBackToLobby = useCallback(() => {
    setActiveGame(null);
  }, []);

  if (activeGame) {
    return (
      <GameView
        joinInfo={activeGame.joinInfo}
        isHost={activeGame.isHost}
        onBackToLobby={handleBackToLobby}
      />
    );
  }

  return <Lobby onJoined={handleJoined} />;
}
