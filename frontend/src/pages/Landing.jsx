export default function Landing() {
  const handleLogin = () => {
    // This should point to your Flask login route
    window.location.href = "http://127.0.0.1:5000/login";
  };

  return (
    <div className="flex flex-col items-center justify-center h-[80vh] text-center">
      <h1 className="text-4xl font-bold mb-6">Discover Your Music Story</h1>
      <p className="mb-8 text-gray-400">
        Connect your Spotify account to explore your listening habits.
      </p>
      <button
        onClick={handleLogin}
        className="bg-green-500 text-black px-6 py-3 rounded-full text-lg font-medium hover:bg-green-400 transition"
      >
        Connect with Spotify
      </button>
    </div>
  );
}
