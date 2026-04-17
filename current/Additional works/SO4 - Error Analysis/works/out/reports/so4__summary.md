# SO4 Error Analysis Summary

## Error rate by system

| system_true         |   n |   error_rate |
|:--------------------|----:|-------------:|
| chatgpt             | 457 |     0.989059 |
| gemini              | 457 |     0.811816 |
| microsoft_translate | 457 |     0.800875 |
| google_translate    | 457 |     0.792123 |

## Error rate by direction

| direction   |   n |   error_rate |
|:------------|----:|-------------:|
| en->fj      | 896 |     0.816964 |
| fj->en      | 932 |     0.878755 |

## Error rate by domain

| domain         |   n |   error_rate |
|:---------------|----:|-------------:|
| bible          | 160 |     0.8      |
| conversational | 312 |     0.692308 |
| definition     | 320 |     0.815625 |
| dictionary     |  80 |     1        |
| idiom          |  80 |     1        |
| legal          | 444 |     0.952703 |
| medical        | 432 |     0.840278 |

## Error rate by sentence type

| sentence_type   | n   | error_rate   |
|-----------------|-----|--------------|
