# Template execution contract

## Reference and verification

- Reference: `C:\Users\sriss\Downloads\Paper Template (1).docx`
- SHA-256: `9053d00389f356af4267c2bec8d46f93627e945a963388dccacfa4f8a156fbbe`
- Package page: A4 portrait, 595.30 by 841.90 pt. Margins are approximately 27/54 pt top (first section differs), 44.65/45.35 pt left/right, 72 pt bottom; header/footer distances 36 pt.
- There are six section-property records. The title and author area uses the source's multi-column section system, the article body is two columns, and the template has a continuous multi-column subsection around its sample figures. Preserve page size, margins, section breaks, column gutters, header/footer, styles, numbering, and package relationships; change the author row from four columns to three.
- Read-only render was attempted with the packaged renderer. It cannot parse the template's Strict OOXML section properties, and LibreOffice is not installed/on PATH. Visual render is unresolved and must be attempted through Word or a compatible renderer before delivery.

## Template components and edit slots

- Body order: title; four author blocks; abstract and keywords; Introduction; Literature review with a summary table; Methodology with architecture and training content; Results and discussions with three sample tables and two sample images; conclusion; references.
- Preserve the source paragraph styles: `papertitle`, `Author`, `Abstract`, `Keywords`, `Heading1` through `Heading5`, `BodyText`, `figurecaption`, `tablehead`, `tablecolhead`, `tablecolsubhead`, `tablecopy`, `references`, and the existing two-column body geometry.
- Replace all template directions, example prose, generic filler, sample equations, unrelated sample citations, sample tables, and unrelated sample illustrations with project content.
- Use three author blocks only; authors are Vangara Sridhar, Suragouni Sahithi, and Vatte Ramya Sri. Affiliation is Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam, Kottayam, Kerala, India. Use the supplied roll numbers and emails only in the author metadata block.
- Project evidence: Training partition 5,600 images (1,400 per class); Testing partition 1,600 images (400 per class). Final test scores: ANN 59.44% accuracy, macro-F1 54.22%; CNN 70.75%, 68.08%; ResNet50 93.69%, 93.53%. Preserve full per-class metrics from `results/*/metrics.json` in the result section. Fine-tuning's best epoch was 15 with validation accuracy 97.26% and loss 0.0609.
- Include explicit task scope (four-class image-level classification, not localization/segmentation or clinical diagnosis), exact split, preprocessing, mathematical comparison, observed limitations, and educational-use caveat.
- Do not invent course name, instructor, submission date, institutional logo, or additional experiment results. Keep these out unless represented by explicit editable blanks in the source.

## Package preservation and fidelity gates

- Author by copying the reference package and modifying only `word/document.xml` plus the existing `word/media/image1.png` and `word/media/image2.png` payloads if using them for relevant figures. Preserve all other parts, relationships, styles, numbering, footnote/endnote parts, footer, and custom XML unless a concrete reference shows they are sample-only. Do not alter the retained source file.
- Confirm source SHA-256 still matches before and after authoring.
- Confirm all sample/template directions and unrelated example reference text are absent from the final document.
- Confirm the final OOXML package opens and includes the three authors, complete sections, measured result tables, and bibliography.
- Render every page and inspect at 100% before delivery. If no compatible renderer is available, report that limitation rather than claiming visual verification.
