export default function App() {
  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      {/* Sol panel — doküman yönetimi */}
      <aside className="w-80 shrink-0 border-r border-zinc-800 bg-zinc-900 p-4 flex flex-col gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">KOBİ RAG</h1>
          <p className="text-xs text-zinc-400">Yerel doküman asistanı</p>
        </div>
        <div className="rounded-lg border border-dashed border-zinc-700 p-6 text-center text-sm text-zinc-400">
          PDF / TXT yükle
        </div>
        <div className="flex-1 overflow-y-auto text-sm text-zinc-500">
          Henüz doküman yok
        </div>
      </aside>

      {/* Sağ panel — sohbet */}
      <main className="flex flex-1 flex-col">
        <div className="flex-1 overflow-y-auto p-6 text-sm text-zinc-500">
          Bir doküman yükleyin ve soru sorun.
        </div>
        <div className="border-t border-zinc-800 p-4">
          <div className="flex gap-2">
            <input
              className="flex-1 rounded-lg bg-zinc-900 border border-zinc-800 px-4 py-2.5 text-sm outline-none focus:border-emerald-600"
              placeholder="Dokümanlarınıza bir soru sorun..."
            />
            <button className="rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-medium hover:bg-emerald-500">
              Sor
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}