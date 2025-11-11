import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="flex justify-between items-center p-4 border-b border-gray-700">
      <h1 className="text-xl font-semibold">Spotify Insights</h1>
      <div className="space-x-4">
        <Link to="/">Home</Link>
        <Link to="/analytics">Analytics</Link>
        {/* Future link: <Link to="/playlist">Playlists</Link> */}
      </div>
    </nav>
  );
}
