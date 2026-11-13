"""Generate synthetic referral and prior-auth PDFs with ground-truth JSON.

Every name, member ID, NPI and payer is invented. NPIs are random 9-digit
numbers with a valid Luhn check digit so validators behave as they would on real
data; they are not looked up against NPPES and any collision with a real NPI is
coincidental. Every page carries a SYNTHETIC footer.

Run:  python -m intake_agent.synth --out data/synthetic --n 40 --seed 7
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .validators import npi_check_digit

PAGE_W, PAGE_H = letter
FOOTER = "SYNTHETIC TEST DOCUMENT - NOT REAL PATIENT DATA - NO PHI"

FIRST = [
    "Avery",
    "Jordan",
    "Riley",
    "Morgan",
    "Casey",
    "Quinn",
    "Harper",
    "Rowan",
    "Elliot",
    "Sage",
    "Parker",
    "Reese",
    "Dakota",
    "Emerson",
    "Finley",
    "Hayden",
    "Kendall",
    "Logan",
]
LAST = [
    "Testwell",
    "Samplesen",
    "Mockridge",
    "Fakeman",
    "Placeholt",
    "Demoro",
    "Synthwood",
    "Exampleton",
    "Fixtureson",
    "Stubbins",
    "Dummond",
    "Proxley",
    "Templeman",
    "Nullberg",
]
PROVIDER_CREDS = ["MD", "DO", "NP"]

PAYERS = {
    "Northwind Health Plan": "NWH",
    "Contoso Care": "CCP",
    "Fabrikam Mutual": "FBM",
    "Tailspin Health Alliance": "THA",
    "Adatum Medical Plan": "ADM",
}

# (requested service, CPT codes, candidate ICD-10 codes)
SERVICES: list[tuple[str, list[str], list[str]]] = [
    ("MRI right knee without contrast", ["73721"], ["M17.11", "M25.561", "S83.241A"]),
    ("Physical therapy evaluation, low complexity", ["97161"], ["M54.50", "M25.511"]),
    ("Screening colonoscopy", ["45378"], ["Z12.11", "Z80.0"]),
    ("Polysomnography, attended in-lab sleep study", ["95810"], ["G47.33", "R06.83"]),
    ("Transthoracic echocardiogram", ["93306"], ["I50.9", "R01.1", "I35.0"]),
    ("New patient cardiology consult", ["99204"], ["I48.91", "R00.2"]),
    ("CT abdomen and pelvis with contrast", ["74177"], ["R10.31", "K57.92"]),
    ("Total knee arthroplasty, left", ["27447"], ["M17.12"]),
    ("Upper GI endoscopy with biopsy", ["43239"], ["K21.9", "R13.10"]),
    ("Lumbar transforaminal epidural steroid injection", ["64483", "64484"], ["M54.16", "M51.16"]),
    ("Dermatology consult, new patient", ["99203"], ["L57.0", "D48.5"]),
    ("Allergy skin testing", ["95004", "95024"], ["J30.9", "L50.9"]),
    ("Diagnostic mammogram, bilateral", ["77066"], ["N63.0", "R92.8"]),
    ("Home sleep apnea test", ["95806"], ["G47.33"]),
    ("Speech-language evaluation", ["92523"], ["R47.01", "I69.320"]),
]

URGENCY_WEIGHTS = {"routine": 0.65, "urgent": 0.25, "stat": 0.10}
PA_PRIORITY = {"routine": "Standard", "urgent": "Expedited", "stat": "Emergent (STAT)"}
LETTER_URGENCY = {
    "routine": "at the next available routine appointment",
    "urgent": "on an urgent basis, ideally within 72 hours",
    "stat": "STAT, same day if at all possible",
}
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Fields that may be left blank on a "partial" document.
DROPPABLE = ["member_id", "referring_npi", "cpt_codes", "patient_dob", "urgency", "payer"]


@dataclass
class Case:
    doc_id: str
    template: str
    document_type: str
    first: str
    last: str
    dob: date
    member_id: str
    payer: str
    provider_first: str
    provider_last: str
    provider_cred: str
    npi: str
    icd10: list[str]
    cpt: list[str]
    service: str
    urgency: str
    request_date: date
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def shown(self, name: str) -> bool:
        return name not in self.dropped

    def truth(self) -> dict:
        def keep(name: str, value):
            if not self.shown(name):
                return [] if isinstance(value, list) else None
            return value

        return {
            "document_type": self.document_type,
            "patient_name": f"{self.first} {self.last}",
            "patient_dob": keep("patient_dob", self.dob.isoformat()),
            "member_id": keep("member_id", self.member_id),
            "payer": keep("payer", self.payer),
            "referring_provider": f"{self.provider_first} {self.provider_last}",
            "referring_npi": keep("referring_npi", self.npi),
            "icd10_codes": sorted(self.icd10),
            "cpt_codes": keep("cpt_codes", sorted(self.cpt)),
            "requested_service": self.service,
            "urgency": keep("urgency", self.urgency),
            "request_date": self.request_date.isoformat(),
        }


def fake_npi(rng: random.Random, valid: bool = True) -> str:
    first_nine = rng.choice("12") + "".join(rng.choice("0123456789") for _ in range(8))
    check = npi_check_digit(first_nine)
    if not valid:
        check = str((int(check) + rng.randint(1, 9)) % 10)
    return first_nine + check


def make_case(rng: random.Random, doc_id: str, template: str) -> Case:
    payer = rng.choice(list(PAYERS))
    service, cpt, icd_pool = rng.choice(SERVICES)
    icd = rng.sample(icd_pool, k=min(len(icd_pool), rng.choice([1, 2])))
    urgency = rng.choices(list(URGENCY_WEIGHTS), weights=list(URGENCY_WEIGHTS.values()))[0]
    return Case(
        doc_id=doc_id,
        template=template,
        document_type=rng.choice(["referral", "prior_auth"]),
        first=rng.choice(FIRST),
        last=rng.choice(LAST),
        dob=date(1940, 1, 1) + timedelta(days=rng.randint(0, 365 * 68)),
        member_id=f"{PAYERS[payer]}{rng.randint(10**8, 10**9 - 1)}",
        payer=payer,
        provider_first=rng.choice(FIRST),
        provider_last=rng.choice(LAST),
        provider_cred=rng.choice(PROVIDER_CREDS),
        npi=fake_npi(rng),
        icd10=icd,
        cpt=list(cpt),
        service=service,
        urgency=urgency,
        request_date=date(2026, 1, 5) + timedelta(days=rng.randint(0, 290)),
    )


def us_date(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def long_date(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def blank_or(case: Case, name: str, text: str, blank: str = "") -> str:
    return text if case.shown(name) else blank


def footer(c: Canvas) -> None:
    c.setFont("Helvetica", 7)
    c.drawCentredString(PAGE_W / 2, 24, FOOTER)


# ---------------------------------------------------------------- templates


def render_form_grid(c: Canvas, case: Case, rng: random.Random) -> None:
    """Template A: clean clinic referral form, 'Label: value' lines."""
    title = (
        "OUTPATIENT REFERRAL FORM"
        if case.document_type == "referral"
        else ("PRIOR AUTHORIZATION REQUEST FORM")
    )
    c.setFont("Helvetica-Bold", 15)
    c.drawString(54, PAGE_H - 60, title)
    c.setFont("Helvetica", 9)
    c.drawString(
        54, PAGE_H - 76, "Synthetic Family Medicine  |  100 Example Ave, Testville  |  Fax 555-0100"
    )
    c.line(54, PAGE_H - 84, PAGE_W - 54, PAGE_H - 84)
    marks = {
        u: "[X]" if case.shown("urgency") and case.urgency == u else "[ ]" for u in URGENCY_WEIGHTS
    }
    rows = [
        ("Patient Name", f"{case.last.upper()}, {case.first.upper()}"),
        ("DOB", blank_or(case, "patient_dob", us_date(case.dob), "____/____/______")),
        ("Member ID", blank_or(case, "member_id", case.member_id, "________________")),
        ("Health Plan", blank_or(case, "payer", case.payer, "________________")),
        ("Referring Provider", f"{case.provider_first} {case.provider_last}, {case.provider_cred}"),
        ("Provider NPI", blank_or(case, "referring_npi", case.npi, "__________")),
        ("Diagnosis (ICD-10)", ", ".join(case.icd10)),
        ("Procedure (CPT)", blank_or(case, "cpt_codes", ", ".join(case.cpt), "_______")),
        ("Service Requested", case.service),
        (
            "Urgency",
            f"{marks['routine']} Routine   {marks['urgent']} Urgent   {marks['stat']} STAT",
        ),
        ("Date of Request", us_date(case.request_date)),
    ]
    y = PAGE_H - 116
    for label, value in rows:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(54, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(190, y, value)
        y -= 26
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(54, y - 10, "Clinical notes: see attached. Please fax scheduling confirmation.")
    footer(c)


def render_boxed_table(c: Canvas, case: Case, rng: random.Random) -> None:
    """Template B: payer-style boxed grid, different labels, ISO dates."""
    c.setFont("Helvetica-Bold", 13)
    head = (
        "UTILIZATION MANAGEMENT - PRECERTIFICATION"
        if case.document_type == "prior_auth"
        else "SPECIALTY CARE REFERRAL"
    )
    c.drawString(54, PAGE_H - 58, head)
    c.setFont("Helvetica", 8)
    c.drawString(
        54, PAGE_H - 70, f"Submitted to: {blank_or(case, 'payer', case.payer, '(plan not listed)')}"
    )
    cells = [
        ("Member Name", f"{case.first} {case.last}"),
        ("Date of Birth", blank_or(case, "patient_dob", case.dob.isoformat())),
        ("Subscriber ID", blank_or(case, "member_id", case.member_id)),
        ("Plan", blank_or(case, "payer", case.payer)),
        ("Ordering Physician", f"Dr. {case.provider_first} {case.provider_last}"),
        ("Ordering NPI", blank_or(case, "referring_npi", case.npi)),
        ("Dx Codes", "  ".join(case.icd10)),
        ("CPT/HCPCS", blank_or(case, "cpt_codes", "  ".join(case.cpt))),
        ("Description of Service", case.service),
        ("Review Priority", blank_or(case, "urgency", PA_PRIORITY[case.urgency])),
        ("Submission Date", case.request_date.isoformat()),
        (
            "Request Type",
            "Prior authorization" if case.document_type == "prior_auth" else "Referral",
        ),
    ]
    col_w, row_h, x0, y0 = (PAGE_W - 108) / 2, 40, 54, PAGE_H - 90
    for i, (label, value) in enumerate(cells):
        col, row = i % 2, i // 2
        x, y = x0 + col * col_w, y0 - row * row_h
        c.rect(x, y - row_h, col_w, row_h)
        c.setFont("Helvetica", 7)
        c.drawString(x + 4, y - 10, label.upper())
        c.setFont("Courier", 10)
        c.drawString(x + 4, y - 27, value)
    footer(c)


def render_letter(c: Canvas, case: Case, rng: random.Random) -> None:
    """Template C: narrative letter. Facts live in prose, not in labeled fields."""
    c.setFont("Times-Bold", 12)
    c.drawString(72, PAGE_H - 64, f"{case.provider_last} Internal Medicine (synthetic)")
    c.setFont("Times-Roman", 11)
    c.drawString(72, PAGE_H - 96, long_date(case.request_date))
    subject = (
        "Letter of medical necessity / prior authorization request"
        if case.document_type == "prior_auth"
        else "Referral for specialty evaluation"
    )
    c.drawString(72, PAGE_H - 124, f"Re: {subject}")
    id_bits = []
    if case.shown("patient_dob"):
        id_bits.append(f"DOB {long_date(case.dob)}")
    if case.shown("member_id"):
        id_bits.append(f"member ID {case.member_id}")
    plan = f" with {case.payer}" if case.shown("payer") else ""
    para = (
        f"To whom it may concern: I am writing regarding my patient {case.first} {case.last}"
        f"{' (' + ', '.join(id_bits) + ')' if id_bits else ''}, insured{plan}. "
        f"I am requesting {case.service.lower()}"
        f"{' ' + LETTER_URGENCY[case.urgency] if case.shown('urgency') else ''}. "
        f"The relevant diagnoses are {' and '.join(case.icd10)}. "
    )
    if case.shown("cpt_codes"):
        para += f"The requested procedure is billed as CPT {', '.join(case.cpt)}. "
    para += (
        "Conservative management has been tried without adequate improvement. "
        "Please contact my office with any questions."
    )
    y = PAGE_H - 156
    for line in simpleSplit(para, "Times-Roman", 11, PAGE_W - 144):
        c.drawString(72, y, line)
        y -= 15
    y -= 30
    c.drawString(72, y, "Sincerely,")
    c.drawString(72, y - 36, f"{case.provider_first} {case.provider_last}, {case.provider_cred}")
    if case.shown("referring_npi"):
        c.drawString(72, y - 51, f"NPI {case.npi}")
    footer(c)


def _scrawl(c: Canvas, x: float, y: float, text: str, rng: random.Random, size: float = 12) -> None:
    """Draw text one glyph at a time with jitter, to imitate handwriting."""
    for ch in text:
        s = size + rng.uniform(-1.0, 1.0)
        c.setFont("Times-Italic", s)
        c.drawString(x, y + rng.uniform(-1.3, 1.3), ch)
        x += stringWidth(ch, "Times-Italic", s) * 1.04 + rng.uniform(0.0, 0.5)


def render_fax_handwritten(c: Canvas, case: Case, rng: random.Random) -> None:
    """Template D: fax cover strip plus a short form filled in by hand."""
    c.setFont("Courier", 8)
    c.drawString(
        36,
        PAGE_H - 30,
        f"FROM: 555-0{rng.randint(100, 199)}  TO: INTAKE 555-0142  "
        f"PG 1/2  {us_date(case.request_date)} 09:{rng.randint(10, 59)}",
    )
    c.setFont("Helvetica-Bold", 12)
    c.drawString(54, PAGE_H - 64, "REFERRAL / AUTH REQUEST  (handwritten)")
    ref = "X" if case.document_type == "referral" else " "
    pa = "X" if case.document_type == "prior_auth" else " "
    c.setFont("Helvetica", 10)
    c.drawString(54, PAGE_H - 84, f"[{ref}] Referral    [{pa}] Prior Auth")
    rows = [
        ("Pt name", f"{case.first} {case.last}"),
        ("DOB", blank_or(case, "patient_dob", us_date(case.dob))),
        ("Ins / ID #", blank_or(case, "member_id", case.member_id)),
        ("Plan", blank_or(case, "payer", case.payer)),
        ("Ref. MD", f"Dr {case.provider_last}, {case.provider_first}"),
        ("NPI #", blank_or(case, "referring_npi", case.npi)),
        ("Dx", " / ".join(case.icd10)),
        ("CPT", blank_or(case, "cpt_codes", " ".join(case.cpt))),
        ("For", case.service),
        (
            "Priority",
            blank_or(
                case, "urgency", case.urgency.upper() if case.urgency != "routine" else "routine"
            ),
        ),
        ("Date", us_date(case.request_date)),
    ]
    y = PAGE_H - 118
    for label, value in rows:
        c.setFont("Helvetica", 9)
        c.drawString(54, y, f"{label}:")
        c.line(130, y - 3, PAGE_W - 72, y - 3)
        if value:
            _scrawl(c, 136, y + 1, value, rng)
        y -= 30
    footer(c)


def _scan_lines(case: Case) -> list[str]:
    """Plain lines for the scanned template, reusing template A wording."""
    urg = case.urgency.upper() if case.shown("urgency") else ""
    kind = "PRIOR AUTHORIZATION REQUEST" if case.document_type == "prior_auth" else "REFERRAL"
    return [
        f"{kind}  (faxed copy)",
        "",
        f"Patient: {case.last}, {case.first}",
        f"DOB: {blank_or(case, 'patient_dob', us_date(case.dob))}",
        f"Member ID: {blank_or(case, 'member_id', case.member_id)}",
        f"Payer: {blank_or(case, 'payer', case.payer)}",
        f"Referring: {case.provider_first} {case.provider_last}, {case.provider_cred}",
        f"NPI: {blank_or(case, 'referring_npi', case.npi)}",
        f"ICD-10: {', '.join(case.icd10)}",
        f"CPT: {blank_or(case, 'cpt_codes', ', '.join(case.cpt))}",
        f"Service: {case.service}",
        f"Urgency: {urg}",
        f"Date: {us_date(case.request_date)}",
        "",
        FOOTER,
    ]


def render_scanned(c: Canvas, case: Case, rng: random.Random) -> None:
    """Template E: image-only page (no text layer), slightly rotated with speckle noise."""
    w, h = 1275, 1650  # 150 dpi letter
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    y = 120
    for line in _scan_lines(case):
        draw.text((110, y), line, fill=rng.randint(10, 60), font=font)
        y += 52
    for _ in range(2500):
        x, yy = rng.randrange(w), rng.randrange(h)
        draw.point((x, yy), fill=rng.randint(80, 200))
    img = img.rotate(rng.uniform(-1.6, 1.6), expand=False, fillcolor=255, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), 0, 0, width=PAGE_W, height=PAGE_H)


TEMPLATES: dict[str, Callable[[Canvas, Case, random.Random], None]] = {
    "form_grid": render_form_grid,
    "boxed_table": render_boxed_table,
    "letter": render_letter,
    "fax_handwritten": render_fax_handwritten,
    "scanned": render_scanned,
}
# Share of documents per template (sums to 1.0).
TEMPLATE_MIX = {
    "form_grid": 0.25,
    "boxed_table": 0.25,
    "letter": 0.2,
    "fax_handwritten": 0.175,
    "scanned": 0.125,
}


def plan_templates(n: int) -> list[str]:
    plan: list[str] = []
    for name, share in TEMPLATE_MIX.items():
        plan += [name] * round(n * share)
    while len(plan) < n:
        plan.append("form_grid")
    return plan[:n]


def generate(out: Path, n: int = 40, seed: int = 7) -> list[Path]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    templates = plan_templates(n)
    rng.shuffle(templates)
    # A few documents carry a mistyped NPI. Ground truth is what is printed.
    bad_npi = set(rng.sample(range(1, n + 1), k=max(1, n // 20)))
    written = []
    for i, template in enumerate(templates, start=1):
        case = make_case(rng, f"doc-{i:03d}", template)
        # About 1 in 5 documents is partial: one or two fields left blank.
        if rng.random() < 0.2:
            case.dropped = rng.sample(DROPPABLE, k=rng.choice([1, 2]))
            case.notes.append(f"partial: {', '.join(case.dropped)} left blank")
        if case.shown("referring_npi") and i in bad_npi:
            case.npi = fake_npi(rng, valid=False)
            case.notes.append("printed NPI fails checksum")
        pdf_path = out / f"{case.doc_id}.pdf"
        c = Canvas(str(pdf_path), pagesize=letter, invariant=1)
        c.setTitle(f"{case.doc_id} (synthetic)")
        TEMPLATES[template](c, case, rng)
        c.showPage()
        c.save()
        meta = {"doc_id": case.doc_id, "template": template, "notes": case.notes}
        truth = {"meta": meta, "fields": case.truth()}
        (out / f"{case.doc_id}.json").write_text(json.dumps(truth, indent=2) + "\n")
        written.append(pdf_path)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    paths = generate(args.out, args.n, args.seed)
    print(f"wrote {len(paths)} PDFs and ground-truth JSON to {args.out}")


if __name__ == "__main__":
    main()
