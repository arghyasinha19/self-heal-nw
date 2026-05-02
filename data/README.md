# Training Data

Place your labelled CSV file in this directory as `training_data.csv`.

## Expected Schema

| Column        | Type   | Description                                      |
|---------------|--------|--------------------------------------------------|
| `description` | string | The raw DNAC alert description text              |
| `category`    | string | One of: `Auto resolving`, `Non-Auto Resolving`   |

## Example

```csv
description,category
"AP went offline and recovered after reboot",Auto resolving
"Switch port security violation detected",Non-Auto Resolving
```

## Notes

- The CSV should contain ~2,500 labelled rows.
- Both `Non-Auto Resolving` and `Non-Auto Resolveing` (typo) are accepted and auto-normalized.
- The file is git-ignored to prevent committing potentially sensitive alert data.
