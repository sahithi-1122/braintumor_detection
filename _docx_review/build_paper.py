from __future__ import annotations

import copy
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree


ROOT = Path(r"C:\Users\sriss\Downloads\Deep Learning")
SOURCE = Path(r"C:\Users\sriss\Downloads\Paper Template (1).docx")
OUTPUT = ROOT / "Brain_Tumor_Classification_Project_Paper.docx"
REVIEW = ROOT / "_docx_review"
W = "http://purl.oclc.org/ooxml/wordprocessingml/main"
R = "http://purl.oclc.org/ooxml/officeDocument/relationships"
NS = {"w": W, "r": R}
Q = lambda name: f"{{{W}}}{name}"


def draw_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.7), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.7)
    ax.axis("off")
    boxes = [
        (0.35, 3.55, 2.0, 1.25, "MRI dataset\nTraining and Testing"),
        (2.8, 3.55, 2.0, 1.25, "Stratified split\n85% / 15%"),
        (5.25, 3.55, 2.0, 1.25, "Preprocessing\nResize and normalize"),
        (7.7, 3.55, 2.0, 1.25, "Augmentation\nTraining only"),
        (1.55, 1.15, 2.0, 1.25, "Feedforward ANN\n64 x 64 input"),
        (4.0, 1.15, 2.0, 1.25, "Custom CNN\n128 x 128 input"),
        (6.45, 1.15, 2.0, 1.25, "ResNet50\n224 x 224 input"),
    ]
    for x, y, width, height, label in boxes:
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#edf3f8", "edgecolor": "#315a78", "linewidth": 1.3},
        )
    arrows = [
        ((2.4, 4.17), (2.8, 4.17)),
        ((4.8, 4.17), (5.25, 4.17)),
        ((7.25, 4.17), (7.7, 4.17)),
        ((8.7, 3.5), (7.45, 2.65)),
        ((8.2, 3.5), (5.0, 2.65)),
        ((7.8, 3.5), (2.55, 2.65)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#52697a"})
    ax.text(5, 0.45, "Held-out Testing partition: accuracy, class-wise metrics, and confusion matrix", ha="center", fontsize=10)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def replace_para(paragraph, text: str, style: str | None = None, plain_run: bool = False) -> None:
    ppr = paragraph.find(Q("pPr"))
    run_props = None
    first_run = paragraph.find(Q("r"))
    if first_run is not None:
        old_rpr = first_run.find(Q("rPr"))
        if old_rpr is not None and not plain_run:
            run_props = copy.deepcopy(old_rpr)
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    if style:
        if ppr is None:
            ppr = etree.Element(Q("pPr"))
            paragraph.insert(0, ppr)
        pstyle = ppr.find(Q("pStyle"))
        if pstyle is None:
            pstyle = etree.Element(Q("pStyle"))
            ppr.insert(0, pstyle)
        pstyle.set(Q("val"), style)
    if text:
        run = etree.SubElement(paragraph, Q("r"))
        if run_props is not None:
            run.append(run_props)
        t = etree.SubElement(run, Q("t"))
        if text[:1].isspace() or text[-1:].isspace():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text


def make_cell(text: str, width: int, header: bool) -> etree._Element:
    tc = etree.Element(Q("tc"))
    tcpr = etree.SubElement(tc, Q("tcPr"))
    tcw = etree.SubElement(tcpr, Q("tcW"))
    tcw.set(Q("w"), str(width))
    tcw.set(Q("type"), "dxa")
    valign = etree.SubElement(tcpr, Q("vAlign"))
    valign.set(Q("val"), "center")
    p = etree.SubElement(tc, Q("p"))
    ppr = etree.SubElement(p, Q("pPr"))
    ps = etree.SubElement(ppr, Q("pStyle"))
    ps.set(Q("val"), "tablecolhead" if header else "tablecopy")
    run = etree.SubElement(p, Q("r"))
    t = etree.SubElement(run, Q("t"))
    t.text = text
    return tc


def make_table(template: etree._Element, rows: list[list[str]], widths: list[int]) -> etree._Element:
    tbl = etree.Element(Q("tbl"), nsmap=template.nsmap)
    tblpr = template.find(Q("tblPr"))
    if tblpr is not None:
        tbl.append(copy.deepcopy(tblpr))
    grid = etree.SubElement(tbl, Q("tblGrid"))
    for width in widths:
        col = etree.SubElement(grid, Q("gridCol"))
        col.set(Q("w"), str(width))
    for row_idx, data in enumerate(rows):
        tr = etree.SubElement(tbl, Q("tr"))
        trpr = etree.SubElement(tr, Q("trPr"))
        if row_idx == 0:
            etree.SubElement(trpr, Q("tblHeader"))
        for text, width in zip(data, widths):
            tr.append(make_cell(text, width, row_idx == 0))
    return tbl


def main() -> None:
    REVIEW.mkdir(exist_ok=True)
    architecture_path = REVIEW / "architecture.png"
    draw_pipeline(architecture_path)
    confusion_path = ROOT / "results" / "resnet" / "confusion_matrix.png"

    with zipfile.ZipFile(SOURCE, "r") as src:
        parts = {item.filename: (item, src.read(item.filename)) for item in src.infolist()}
    document = etree.fromstring(parts["word/document.xml"][1])
    body = document.find(".//" + Q("body"))
    children = list(body)

    def p(index: int):
        item = children[index]
        if item.tag != Q("p"):
            raise ValueError(f"Body index {index} is not a paragraph")
        return item

    title = "Brain Tumor MRI Classification Using Deep Learning"
    replace_para(p(0), title)

    author_blocks = [
        (3, "Vangara Sridhar", "2023BCS0106", "vangara23bcs106@iiitkottaym.ac.in"),
        (12, "Suragouni Sahithi", "2023BCS0070", "suragou23bcs70@iiitkottaym.ac.in"),
        (20, "Vatte Ramya Sri", "2023BCS0208", "vatte23bcs208@iiitkottaym.ac.in"),
    ]
    for start, name, roll, email in author_blocks:
        replace_para(p(start), f"{name} ({roll})")
        replace_para(p(start + 1), "Department of Computer Science and Engineering")
        replace_para(p(start + 2), "")
        replace_para(p(start + 3), "Indian Institute of Information Technology Kottayam")
        replace_para(p(start + 4), "")
        replace_para(p(start + 5), "Kottayam, Kerala, India")
        replace_para(p(start + 6), email)

    # Use three author columns while preserving the template's section break.
    section_after_authors = p(37).find("w:pPr/w:sectPr", NS)
    if section_after_authors is not None:
        cols = section_after_authors.find(Q("cols"))
        if cols is not None:
            cols.set(Q("num"), "3")

    replace_para(
        p(41),
        "Abstract—This paper compares a feedforward artificial neural network, a custom convolutional neural network, and an ImageNet pretrained ResNet50 for four class brain MRI image classification. The local dataset contains 5,600 training images and 1,600 test images, balanced across glioma, meningioma, no tumor, and pituitary classes. We split the training partition into 4,760 training and 840 validation images and evaluate on the supplied test partition. ResNet50 achieved 93.69 percent test accuracy and 93.53 percent macro F1, compared with 70.75 percent and 68.08 percent for the custom CNN and 59.44 percent and 54.22 percent for the ANN. Results are educational and are not clinical validation.",
    )
    replace_para(p(42), "Keywords—brain MRI, tumor classification, artificial neural network, convolutional neural network, transfer learning")
    replace_para(p(43), "Introduction")
    replace_para(
        p(44),
        "Brain MRI interpretation involves visual patterns that vary in size, shape, and appearance. This project studies an image classification task: assign each MRI image to one of four dataset labels, glioma, meningioma, no tumor, or pituitary tumor. The task does not locate or segment a tumor and does not provide a medical diagnosis.\n\nWe compare a flattened feedforward network, a custom CNN, and an ImageNet pretrained ResNet50. The comparison examines held out accuracy, macro averaged metrics, class wise errors, and model architecture. The goal is to show how spatial feature extraction and transfer learning affect this dataset, using a fixed test partition and reproducible split seed.",
    )
    replace_para(p(45), "Literature Review")
    replace_para(p(46), "Deep Learning Methods for MRI Classification")
    replace_para(
        p(47),
        "Earlier image classification systems often combined manually designed image features with conventional classifiers. CNNs learn local filters from labeled images and build hierarchical features, and surveys describe their broad use in medical image analysis [2]. Brain tumor image classification has also been studied with CNN architectures [3], [4].",
    )
    replace_para(
        p(48),
        "Residual networks use identity shortcuts to improve optimization of deep models [7]. Transfer learning initializes a model from a source task before adapting it to a target dataset [8], [9]. Vision Transformers and self supervised learning are related current directions [11]. These studies motivate our model selection; their reported scores are not directly comparable with ours because datasets and protocols differ.",
    )
    replace_para(p(49), "TABLE I. REPRESENTATIVE METHODS IN RELATED WORK", style="tablehead")
    table_nodes = [node for node in children if node.tag == Q("tbl")]
    table1 = make_table(
        table_nodes[0],
        [
            ["Approach", "Representative work", "Relevance to this project"],
            ["CNN medical imaging", "Litjens et al. [2]", "Learned spatial features"],
            ["Brain MRI CNN", "Abiwinanda et al. [3]", "Related image classification task"],
            ["Residual learning", "He et al. [7]", "Deep ResNet50 backbone"],
            ["Transfer learning", "Pan and Yang [8]", "Pretrained model adaptation"],
        ],
        [1000, 1500, 2360],
    )

    replace_para(p(52), "Methodology")
    replace_para(
        p(53),
        "Dataset and split. We used the supplied Kaggle Brain Tumor MRI Dataset [1]. The local copy contains 1,400 images per class in Training (5,600 total) and 400 images per class in Testing (1,600 total). The no tumor folder is named notumor and is mapped by the loader to the canonical label no_tumor. Training was stratified into 4,760 training and 840 validation images using seed 42. The official Testing partition was held out until final evaluation.",
    )
    replace_para(
        p(54),
        "Preprocessing. OpenCV loads images, converts BGR to RGB, and resizes them to each architecture's input size. ANN and CNN inputs are scaled to [0, 1]. ResNet50 uses its Keras ImageNet preprocessing, which reorders channels and subtracts the ImageNet channel means. Training only uses random horizontal flips, rotations of approximately 15 degrees, zoom, and brightness changes. Validation and test images are not augmented.",
    )
    replace_para(
        p(55),
        "Feedforward ANN. Images are resized to 64 by 64 pixels and flattened to 12,288 features. The network has dense layers with 512 and 256 ReLU units, dropout of 0.4, and a four unit softmax output. It has 6,424,324 parameters. Flattening does not preserve explicit local pixel neighborhoods.",
    )
    replace_para(
        p(56),
        "Custom CNN. The 128 by 128 input passes through three convolution blocks with 32, 64, and 128 filters, 3 by 3 kernels, batch normalization, ReLU, and max pooling. Global average pooling feeds a 256 unit dense layer, dropout, and a four class softmax head. The model has approximately 127,524 parameters.",
    )
    replace_para(
        p(57),
        "ResNet50 transfer learning. A ResNet50 backbone pretrained on ImageNet receives 224 by 224 inputs. Global average pooling and a 256 unit dense classification head produce four probabilities. Phase one freezes the backbone; phase two fine tunes the top 30 layers with learning rate 0.00001. The complete model has approximately 24.1 million parameters.",
    )
    replace_para(p(59), "Fig. 1. Training and evaluation workflow.", style="figurecaption")
    replace_para(p(60), "Architecture and Mathematical Basis", style="Heading2")
    replace_para(
        p(61),
        "A dense layer computes z = W x + b followed by an activation. The ANN's first dense layer connects every input pixel to every hidden unit. A convolution instead applies a shared kernel to local neighborhoods. A 3 by 3 convolution from C input channels to F filters uses 9CF kernel weights, independent of image width and height; this weight sharing explains the much smaller CNN parameter count. Residual blocks compute y = F(x) + x, providing an identity path through the ResNet. All models use a four class softmax and categorical cross entropy.",
    )
    replace_para(p(62), "Training and Evaluation", style="Heading2")
    replace_para(p(63), "Optimizer: Adam with initial learning rate 0.001; ResNet fine tuning uses 0.00001 [10].", style="bulletlist")
    replace_para(p(64), "Maximum 50 epochs for the initial models; early stopping monitors validation loss with patience 10 and restores best weights.", style="bulletlist")
    replace_para(p(65), "ResNet fine tuning runs for at most 20 epochs with patience 5; the saved final model restores the best validation checkpoint.", style="bulletlist")
    replace_para(p(66), "Metrics: accuracy, macro precision, macro recall, macro F1, confusion matrix, and class wise precision, recall, and F1.", style="bulletlist")
    replace_para(p(78), "The final evaluation uses the same 1,600 untouched Testing images for all three models. Each class contributes 400 images.")

    replace_para(p(79), "Results and Discussion")
    replace_para(
        p(80),
        "Table II shows that ResNet50 achieved the best performance on the supplied test partition, with 93.69 percent accuracy and 0.9353 macro F1. The custom CNN achieved 70.75 percent accuracy and 0.6808 macro F1. The ANN achieved 59.44 percent accuracy and 0.5422 macro F1. The comparison is consistent with the benefit of spatial inductive biases and pretrained initialization in this experiment, but it does not isolate their causal effects.",
    )
    replace_para(
        p(81),
        "During ResNet50 fine tuning, the best validation loss was 0.0609 at epoch 15, with validation accuracy 97.26 percent. Early stopping restored those weights after the remaining patience window. Test accuracy was lower at 93.69 percent, a reminder that validation performance is not a substitute for held out evaluation.",
    )
    replace_para(
        p(82),
        "Macro F1 is the unweighted mean of the class F1 scores. For each class, F1 = 2 times precision times recall divided by precision plus recall. The confusion matrix in Fig. 2 shows that the largest ResNet50 error group was glioma predicted as meningioma: 61 of 400 glioma images.",
    )
    replace_para(p(83), "F1 = 2 * precision * recall / (precision + recall)", style="BodyText", plain_run=True)
    replace_para(
        p(84),
        "The ANN had only 6.00 percent meningioma recall. The CNN had 37.25 percent glioma recall despite high precision for that class. ResNet50 improved these recalls to 96.50 percent for meningioma and 78.50 percent for glioma, while its no tumor and pituitary recalls were 100.00 and 99.75 percent.",
    )
    replace_para(p(86), "Held Out Test Performance", style="Heading2")
    replace_para(
        p(87),
        "All reported test metrics use 1,600 images, with equal support of 400 images per class. Table II reports aggregate metrics, and Table III gives class wise precision, recall, and F1. Confusion matrices and training curves are saved under results/.",
    )
    replace_para(p(88), "Class Wise Results", style="Heading3")
    replace_para(
        p(89),
        "The ANN's low meningioma recall indicates that its overall accuracy hides a serious class specific weakness. Its macro F1 of 0.5422 is lower than its accuracy because macro averaging gives each class equal weight.",
        style="BodyText",
    )
    replace_para(
        p(90),
        "The CNN strongly favored pituitary and no tumor recall but missed many glioma and meningioma examples. Its 0.6808 macro F1 better reflects this uneven behavior than accuracy alone.",
        style="BodyText",
    )
    replace_para(
        p(91),
        "ResNet50 achieved the highest macro F1, 0.9353. Glioma remained the most difficult class for this model, with 0.7850 recall; its remaining classes had recalls from 0.9650 to 1.0000.",
        style="BodyText",
    )
    replace_para(
        p(92),
        "Exact SHA 256 comparison found no identical image files between the local Training and Testing partitions. No patient identifiers were available, so this does not establish patient level independence or rule out near duplicates.",
        style="BodyText",
    )
    replace_para(p(97), "Fig. 2. ResNet50 confusion matrix on the 1,600 image test partition.", style="figurecaption")
    replace_para(p(100), "Interpretation and Limitations", style="Heading2")
    replace_para(
        p(101),
        "The CNN uses local connectivity and shared filters, giving it a strong architectural fit for image inputs while using far fewer parameters than the ANN. ResNet50 adds a much deeper residual feature extractor initialized with ImageNet weights. In this comparison, the transfer learning model performed best, but it also has far more total parameters and uses a different input resolution and preprocessing pipeline.",
    )
    replace_para(
        p(102),
        "This is a single dataset experiment. The test set is balanced and comes from the same provided dataset release; generalization to other hospitals, scanners, MRI sequences, or patient populations has not been tested. The models classify whole images and do not provide tumor localization, segmentation, calibrated uncertainty, or clinical decision support.",
    )
    replace_para(
        p(103),
        "The reported scores should be interpreted as educational benchmark results. Before any clinical use, models would require independent patient level validation, broader data, careful bias and calibration analysis, expert oversight, and applicable regulatory review.",
    )
    replace_para(p(104), "TABLE II. HELD OUT TEST PARTITION PERFORMANCE", style="tablehead")
    table2 = make_table(
        table_nodes[1],
        [
            ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1"],
            ["ANN / MLP", "0.5944", "0.6775", "0.5944", "0.5422"],
            ["Custom CNN", "0.7075", "0.7491", "0.7075", "0.6808"],
            ["ResNet50", "0.9369", "0.9417", "0.9369", "0.9353"],
        ],
        [1000, 1000, 950, 950, 960],
    )
    replace_para(p(107), "Class Wise Test Metrics", style="Heading2")
    replace_para(p(108), "Precision, recall, and F1 are computed separately for each class on the same held out Testing partition.", style="BodyText")
    replace_para(p(109), "TABLE III. CLASS WISE PRECISION RECALL AND F1", style="tablehead")
    table3 = make_table(
        table_nodes[2],
        [
            ["Model", "Class", "Precision", "Recall", "F1"],
            ["ANN", "Glioma", "0.4333", "0.8525", "0.5746"],
            ["ANN", "Meningioma", "0.7742", "0.0600", "0.1114"],
            ["ANN", "No tumor", "0.8000", "0.7500", "0.7742"],
            ["ANN", "Pituitary", "0.7027", "0.7150", "0.7088"],
            ["CNN", "Glioma", "0.9868", "0.3725", "0.5408"],
            ["CNN", "Meningioma", "0.6133", "0.5075", "0.5554"],
            ["CNN", "No tumor", "0.7111", "0.9600", "0.8170"],
            ["CNN", "Pituitary", "0.6851", "0.9900", "0.8098"],
            ["ResNet50", "Glioma", "0.9937", "0.7850", "0.8771"],
            ["ResNet50", "Meningioma", "0.8616", "0.9650", "0.9104"],
            ["ResNet50", "No tumor", "0.9479", "1.0000", "0.9732"],
            ["ResNet50", "Pituitary", "0.9638", "0.9975", "0.9803"],
        ],
        [850, 1250, 920, 920, 920],
    )
    replace_para(
        p(112),
        "Class support is 400 images for each class. Rounding may cause displayed values to differ slightly from the full precision metrics in results/.",
        style="BodyText",
    )
    replace_para(p(114), "Conclusion and Future Work")
    replace_para(
        p(115),
        "We compared an ANN, a custom CNN, and ResNet50 for four class brain MRI image classification. On the supplied 1,600 image test partition, ResNet50 achieved the highest accuracy and macro F1, followed by the custom CNN and ANN. The CNN uses convolutional weight sharing to model local spatial features with fewer parameters than the flattened ANN, while residual connections and ImageNet initialization provide a deeper transfer learning model. The class wise results also expose important weaknesses that overall accuracy alone would hide. Future work should test patient level and external validation, add localization and interpretability, and evaluate other MRI sequences. This system is for education only and must not be used for clinical diagnosis.",
    )
    replace_para(p(117), "References", style="Heading5")

    references = [
        "[1] M. Nickparvar, Brain Tumor MRI Dataset, Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "[2] G. Litjens et al., A survey on deep learning in medical image analysis, Medical Image Analysis, vol. 42, pp. 60-88, 2017.",
        "[3] N. Abiwinanda, M. Hanif, S. T. Hesaputra, A. Handayani, and T. R. Mengko, Brain tumor classification using convolutional neural network, in Proc. IFMBE, vol. 68, 2019, pp. 183-189.",
        "[4] H. H. Sultan, N. M. Salem, and W. Al-Atabany, Multi-classification of brain tumor images using deep neural network, IEEE Access, vol. 7, pp. 69215-69225, 2019.",
        "[5] A. Krizhevsky, I. Sutskever, and G. E. Hinton, ImageNet classification with deep convolutional neural networks, in Advances in Neural Information Processing Systems, vol. 25, 2012, pp. 1097-1105.",
        "[6] K. Simonyan and A. Zisserman, Very deep convolutional networks for large-scale image recognition, in Proc. ICLR, 2015.",
        "[7] K. He, X. Zhang, S. Ren, and J. Sun, Deep residual learning for image recognition, in Proc. IEEE CVPR, 2016, pp. 770-778.",
        "[8] S. J. Pan and Q. Yang, A survey on transfer learning, IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, 2010.",
        "[9] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, How transferable are features in deep neural networks?, in Advances in Neural Information Processing Systems, vol. 27, 2014.",
        "[10] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, in Proc. ICLR, 2015.",
        "[11] A. Dosovitskiy et al., An image is worth 16x16 words: Transformers for image recognition at scale, in Proc. ICLR, 2021.",
        "[12] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature Methods, vol. 18, pp. 203-211, 2021.",
    ]
    for idx, text in zip(range(118, 130), references):
        replace_para(p(idx), text, style="references")

    # Remove the unused fourth author block and all remaining IEEE tutorial text.
    remove_indices = set(range(28, 35)) | set(range(67, 78)) | set(range(130, 134))
    for idx in sorted(remove_indices, reverse=True):
        body.remove(children[idx])

    # Replace the three example tables with compact project tables in the same positions.
    for old, new in zip(table_nodes, [table1, table2, table3]):
        old.getparent().replace(old, new)

    # Update the two template image slots with project-specific figures.
    parts["word/media/image1.png"] = (parts["word/media/image1.png"][0], architecture_path.read_bytes())
    parts["word/media/image2.png"] = (parts["word/media/image2.png"][0], confusion_path.read_bytes())

    # Keep the confusion matrix undistorted in its inline drawing.
    for paragraph in body.findall("w:p", NS):
        for inline in paragraph.findall(".//{http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing}inline", NS):
            blip = inline.find(".//{http://purl.oclc.org/ooxml/drawingml/main}blip", NS)
            if blip is not None and blip.get(f"{{{R}}}embed") == "rId10":
                width = int(2.25 * 914400)
                height = int(width * 900 / 1050)
                extent = inline.find("{http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing}extent")
                if extent is not None:
                    extent.set("cx", str(width)); extent.set("cy", str(height))
                xfrm_ext = inline.find(".//{http://purl.oclc.org/ooxml/drawingml/main}ext")
                if xfrm_ext is not None:
                    xfrm_ext.set("cx", str(width)); xfrm_ext.set("cy", str(height))

    parts["word/document.xml"] = (parts["word/document.xml"][0], etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True))

    footer = etree.fromstring(parts["word/footer1.xml"][1])
    for paragraph in footer.findall(".//w:p", NS):
        replace_para(paragraph, "")
    parts["word/footer1.xml"] = (parts["word/footer1.xml"][0], etree.tostring(footer, xml_declaration=True, encoding="UTF-8", standalone=True))

    core = etree.fromstring(parts["docProps/core.xml"][1])
    dct = {"dc": "http://purl.org/dc/elements/1.1/", "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"}
    for tag, value in [("title", title), ("subject", "Deep learning project paper for brain tumor MRI image classification"), ("creator", "Vangara Sridhar; Suragouni Sahithi; Vatte Ramya Sri")]:
        node = core.find(f"dc:{tag}", dct) if tag != "subject" else core.find("dc:subject", dct)
        if node is None:
            node = etree.SubElement(core, f"{{{dct['dc']}}}{tag}")
        node.text = value
    last = core.find("cp:lastModifiedBy", dct)
    if last is not None:
        last.text = "Vangara Sridhar, Suragouni Sahithi, Vatte Ramya Sri"
    parts["docProps/core.xml"] = (parts["docProps/core.xml"][0], etree.tostring(core, xml_declaration=True, encoding="UTF-8", standalone=True))

    app = etree.fromstring(parts["docProps/app.xml"][1])
    appns = {"a": "http://purl.oclc.org/ooxml/officeDocument/extendedProperties"}
    company = app.find("a:Company", appns)
    if company is not None:
        company.text = "Indian Institute of Information Technology Kottayam"
    parts["docProps/app.xml"] = (parts["docProps/app.xml"][0], etree.tostring(app, xml_declaration=True, encoding="UTF-8", standalone=True))

    with zipfile.ZipFile(OUTPUT, "w") as out:
        for name, (info, data) in parts.items():
            out.writestr(info, data)
    print(OUTPUT)


if __name__ == "__main__":
    main()
