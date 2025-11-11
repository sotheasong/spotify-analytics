import { useEffect, useState } from "react";

export default function Analytics() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/top-artists")
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="p-8">
      <h2 className="text-3xl font-semibold mb-6">Your Listening Analytics</h2>

      {!data ? (
        <p>Loading your data...</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.items?.map((artist) => (
            <div
              key={artist.id}
              className="bg-gray-800 p-4 rounded-lg flex items-center space-x-4"
            >
              <img
                src={artist.images[0]?.url}
                alt={artist.name}
                className="w-16 h-16 rounded-full object-cover"
              />
              <div>
                <h3 className="text-lg font-medium">{artist.name}</h3>
                <p className="text-gray-400">{artist.genres.join(", ")}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
