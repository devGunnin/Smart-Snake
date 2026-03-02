import { useCallback, useState } from "react";
import GameView from "./components/GameView";
import Lobby from "./components/Lobby";
import type { JoinResponse } from "./types/game";

export default function App() {
  const [activeGame, setActiveGame] = useState<JoinResponse | null>(null);

  const handleJoined = useCallback((joinInfo: JoinResponse) => {
    setActiveGame(joinInfo);
  }, []);

  const handleBackToLobby = useCallback(() => {
    setActiveGame(null);
  }, []);

  if (activeGame) {
    return (
      <GameView
        joinInfo={activeGame}
        onBackToLobby={handleBackToLobby}
      />
    );
  }

  return <Lobby onJoined={handleJoined} />;
}
