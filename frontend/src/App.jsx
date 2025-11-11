import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Landing from "./pages/Landing";
import Analytics from "./pages/Analytics";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#0e0e0e] text-white flex flex-col">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/analytics" element={<Analytics />} />
            {/* Future route */}
            {/* <Route path="/playlist" element={<Playlist />} /> */}
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
