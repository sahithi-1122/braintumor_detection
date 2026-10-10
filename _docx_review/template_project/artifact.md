# Template execution contract

## Reference and evidence

- Reference: `C:\Users\sriss\Downloads\template_project.docx`
- SHA-256: `c2b5db4d4dc7e8fdc7f71941f336910ce9ed2315abac2ab5cd5d8a6263604dbc` (462,377 bytes)
- Evidence: `template-style-evidence.json`; section audit reports six sections. Template rendering was attempted to `reference-render` but failed because LibreOffice (`soffice.exe`) is unavailable. The source is an A4 IEEE-like conference-paper layout.
- Retained source must remain unchanged. Target output is a separate DOCX in the writable project workspace.

## Page system and typography

- A4 portrait, 8.27 × 11.69 in. Outer margins generally L/R 0.62–0.63 in, top 0.375–0.75 in, bottom 1.00 in; header/footer distances 0.5 in.
- Six source sections: title/author front matter, author-column section, two-column body sections and final section. Preserve the source section properties, linked headers/footers, page numbering behavior and styles. Author area is a four-column source pattern; change to three columns for this three-author team and remove the unused fourth author block.
- Existing paragraph roles to reuse: `paper title`, `Author`, `Abstract`, `Keywords`, `Heading 1` through `Heading 5`, `Body Text`, `bullet list`, `equation`, `figure caption`, `table head`, and `references`. Source fonts are Times New Roman/Symbol, with IEEE-style compact two-column body text. Keep template type sizes, columns and rhythm; do not apply a new style system.

## Editable slots and intended content flow

1. Title and three author blocks (name plus roll, department, institution, location, email). Delete the fourth block.
2. Abstract and keywords.
3. Introduction and Literature Review, with a concise related-work comparison table.
4. Methodology: dataset, split/preprocessing, ANN, CNN, ResNet50 transfer learning, training configuration and architectural/mathematical rationale. Reuse first source image slot for project architecture; update its caption.
5. Results and Discussion: report actual supplied test-set results for ANN, CNN and ResNet50; include aggregate comparison table, ResNet per-class table, and use second source image slot for the ResNet confusion matrix. Update associated captions. Discuss errors and limitations without clinical claims.
6. Conclusion and Future Work; references in the template's IEEE reference style.

## Tables, figures, and package preservation

- Three existing 4-column, 3-row tables are replaceable placeholders. Resize rows/columns as needed for a literature snapshot, aggregate model comparison, and ResNet per-class metrics. Retain table style, borders, and source table paragraph styles.
- Two existing inline image relationships are placeholders and may be replaced in place with the project's architecture diagram and ResNet50 confusion matrix. Keep inline anchors and their caption positions.
- Preserve styles, numbering, section properties, headers/footers, theme, document settings, relationships and unrelated package parts. Only the main document content and the two existing image payloads may change, along with document metadata if needed.

## Facts to use

- Project: four-class brain MRI image classification (glioma, meningioma, no tumor, pituitary) on the Masoud Nickparvar Kaggle Brain Tumor MRI Dataset.
- Local evaluation artifacts report 5,600 training images (1,400/class), 1,600 held-out test images (400/class), with 4,760 training and 840 validation images from a stratified 85/15 split of Training. The folder alias `notumor` maps to canonical `no_tumor`.
- Test accuracy / macro-F1: ANN 0.5944 / 0.5422; CNN 0.7075 / 0.6808; ResNet50 0.9369 / 0.9353. These values must match checked-in JSON/CSV outputs.
- Fine-tuning best checkpoint: epoch 15, validation loss 0.0609 and validation accuracy 97.26%; early stopping at epoch 20.
- The project is educational, image-level classification only; it is not clinically validated.

## Fidelity and content gates

- All instructional boilerplate and sample paper text must be removed. Keep the source hierarchy and IEEE-like appearance, but do not treat generic placeholder wording (including a sample reference count) as report content.
- No unsupported clinical claims, invented findings, or fabricated citations. The reference section includes the template's stated minimum of 23 relevant sources, each cited in the paper.
- Final package must open as a valid DOCX, contain no template placeholders, preserve source section count/page geometry, and be visually checked if a renderer is available. Source SHA-256 must remain unchanged.
