export const GRID_LAYOUT_VERSION = 2;
export const DEFAULT_GRID_COLUMNS = 12;
export const DEFAULT_GRID_ROWS = 12;

const isRecord = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const toRoundedNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.round(parsed) : null;
};

const toInteger = (value, fallback = null) => {
  const parsed = toRoundedNumber(value);
  if (parsed === null) return fallback;
  return parsed;
};

const toPositiveInt = (value, fallback = 1) => {
  const parsed = toRoundedNumber(value);
  if (parsed === null) return fallback;
  return Math.max(1, parsed);
};

const inferType = (id, fallback = 'track') => {
  if (typeof id === 'string' && id.startsWith('drum_')) return 'drum';
  return fallback;
};

const compareLayoutEntries = ([idA, itemA], [idB, itemB]) => {
  const posA = toRoundedNumber(itemA?.position);
  const posB = toRoundedNumber(itemB?.position);

  if (posA !== null || posB !== null) {
    if (posA === null) return 1;
    if (posB === null) return -1;
    if (posA !== posB) return posA - posB;
  }

  const yA = toInteger(itemA?.y ?? itemA?.row, 0);
  const yB = toInteger(itemB?.y ?? itemB?.row, 0);
  if (yA !== yB) return yA - yB;

  const xA = toInteger(itemA?.x ?? itemA?.column, 0);
  const xB = toInteger(itemB?.x ?? itemB?.column, 0);
  if (xA !== xB) return xA - xB;

  return String(idA).localeCompare(String(idB));
};

export const isGridArrangementV2 = (arrangement) =>
  isRecord(arrangement) && isRecord(arrangement.items);

export const normalizeGridArrangement = (
  arrangement,
  {
    defaultColumns = DEFAULT_GRID_COLUMNS,
    defaultRows = DEFAULT_GRID_ROWS,
  } = {},
) => {
  if (isGridArrangementV2(arrangement)) {
    const items = {};

    for (const [id, rawItem] of Object.entries(arrangement.items)) {
      if (!isRecord(rawItem) || rawItem.isEmpty) continue;

      const x = toInteger(rawItem.x ?? rawItem.column);
      const y = toInteger(rawItem.y ?? rawItem.row);
      if (x === null || y === null) continue;

      const w = toPositiveInt(rawItem.w, 1);
      const h = toPositiveInt(rawItem.h, 1);

      items[id] = {
        x,
        y,
        w,
        h,
        type: rawItem.type || inferType(id),
      };

      const position = toRoundedNumber(rawItem.position);
      if (position !== null) {
        items[id].position = position;
      }
    }

    return {
      version: GRID_LAYOUT_VERSION,
      columns: toPositiveInt(arrangement.columns, defaultColumns),
      rows: toPositiveInt(arrangement.rows, defaultRows),
      items,
    };
  }

  if (!isRecord(arrangement)) {
    return {
      version: GRID_LAYOUT_VERSION,
      columns: defaultColumns,
      rows: defaultRows,
      items: {},
    };
  }

  const items = {};
  let maxColumnEnd = 0;
  let maxRowEnd = 0;

  for (const [id, rawItem] of Object.entries(arrangement)) {
    if (!isRecord(rawItem) || rawItem.isEmpty) continue;

    const x = toInteger(rawItem.column ?? rawItem.x);
    const y = toInteger(rawItem.row ?? rawItem.y);
    if (x === null || y === null) continue;

    const w = toPositiveInt(rawItem.w, 1);
    const h = toPositiveInt(rawItem.h, 1);
    maxColumnEnd = Math.max(maxColumnEnd, x + w);
    maxRowEnd = Math.max(maxRowEnd, y + h);

    items[id] = {
      x,
      y,
      w,
      h,
      type: rawItem.type || inferType(id),
    };

    const position = toRoundedNumber(rawItem.position);
    if (position !== null) {
      items[id].position = position;
    }
  }

  return {
    version: GRID_LAYOUT_VERSION,
    columns: maxColumnEnd > 0 ? maxColumnEnd : defaultColumns,
    rows: maxRowEnd > 0 ? maxRowEnd : defaultRows,
    items,
  };
};

export const toLegacyGridArrangement = (arrangement) => {
  const normalized = normalizeGridArrangement(arrangement);
  const orderedEntries = Object.entries(normalized.items).sort(
    compareLayoutEntries,
  );

  return Object.fromEntries(
    orderedEntries.map(([id, item], index) => [
      id,
      {
        row: item.y,
        column: item.x,
        position: toRoundedNumber(item.position) ?? index,
        type: item.type || inferType(id),
        w: item.w,
        h: item.h,
      },
    ]),
  );
};

export const hasGridArrangement = (arrangement) =>
  Object.keys(toLegacyGridArrangement(arrangement)).length > 0;

export const getGridArrangementOverflow = (
  arrangement,
  {
    defaultColumns = DEFAULT_GRID_COLUMNS,
    defaultRows = DEFAULT_GRID_ROWS,
  } = {},
) => {
  if (!isGridArrangementV2(arrangement)) {
    return null;
  }

  const normalized = normalizeGridArrangement(arrangement, {
    defaultColumns,
    defaultRows,
  });
  const columns = normalized.columns;
  const rows = normalized.rows;

  for (const [id, item] of Object.entries(normalized.items)) {
    if (item.x < 0 || item.y < 0) {
      return {
        id,
        item,
        columns,
        rows,
      };
    }

    if (item.x + item.w > columns || item.y + item.h > rows) {
      return {
        id,
        item,
        columns,
        rows,
      };
    }
  }

  return null;
};

export const getGridArrangementBounds = (arrangement) => {
  const normalized = normalizeGridArrangement(arrangement);
  const items = Object.values(normalized.items);

  if (items.length === 0) {
    return {
      maxRow: 0,
      maxCol: 0,
      rowCount: 1,
      columnCount: 1,
    };
  }

  const maxRow = Math.max(...items.map((item) => item.y + item.h - 1));
  const maxCol = Math.max(...items.map((item) => item.x + item.w - 1));

  return {
    maxRow,
    maxCol,
    rowCount: maxRow + 1,
    columnCount: Math.max(normalized.columns, maxCol + 1),
  };
};
