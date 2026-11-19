import { businessModeLabel as modeLabel } from "../domain/business-modes.ts";
import type { NV200SheetData } from "../domain/nv200-table";

export default function ModeSheetBar({
  allSheets,
  selectedSheetId,
  onSelectSheetId,
}: {
  allSheets: NV200SheetData[];
  selectedSheetId: string;
  onSelectSheetId: (sheetId: string) => void;
}) {
  return (
    <div className="nv200-sheet-bar" aria-label="业务模式选择">
      {allSheets.map((sheet) => {
        const isActive = sheet.sheetId === selectedSheetId;
        return (
          <button
            key={sheet.sheetId}
            className={`nv200-sheet-pill ${isActive ? "active" : ""}`}
            onClick={() => onSelectSheetId(sheet.sheetId)}
            aria-pressed={isActive}
          >
            <span>{modeLabel(sheet.sheetId)}</span>
            <span className="sheet-stat">42 项</span>
          </button>
        );
      })}
    </div>
  );
}
