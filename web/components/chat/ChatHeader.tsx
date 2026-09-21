interface ChatHeaderProps {
  onOpenSettings: () => void;
  onClearHistory: () => void;
}

export function ChatHeader({ onOpenSettings, onClearHistory }: ChatHeaderProps) {
  return (
    <header className="mb-4 flex items-center justify-between">
      <div className="flex items-center">
        <h1 className="text-lg font-semibold">OpenPoke 🌴</h1>
      </div>
      <div className="flex items-center gap-2">
        <Link className="rounded-md border border-blue-200 px-3 py-2 text-sm text-blue-700 hover:bg-blue-50" href="/lab">Evaluation Lab</Link>
        <button
          className="rounded-md border border-gray-200 px-3 py-2 text-sm hover:bg-gray-50"
          onClick={onOpenSettings}
        >
          Settings
        </button>
        <button
          className="rounded-md border border-gray-200 px-3 py-2 text-sm hover:bg-gray-50"
          onClick={onClearHistory}
        >
          Clear
        </button>
      </div>
    </header>
  );
}
import Link from 'next/link';
