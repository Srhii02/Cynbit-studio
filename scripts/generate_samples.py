"""Generate synthetic sample resumes (PDF/DOCX/TXT) and job descriptions (TXT).

All people, companies, emails and phone numbers are fictional (example.com, 555 numbers).
Run:  python scripts/generate_samples.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "samples" / "resumes"
JOB_DIR = ROOT / "samples" / "jobs"

RESUMES: dict[str, str] = {
"ananya_rao_ml_student": """Ananya Rao
ananya.rao@example.com | +91 55500 11223 | linkedin.com/in/ananya-rao-demo | github.com/ananya-rao-demo
Hyderabad, India

SUMMARY
Final-year Computer Science student focused on machine learning and NLP. Built and deployed three end-to-end ML
projects, including a semantic search engine over 20k documents.

EDUCATION
B.Tech in Computer Science and Engineering, Vardhaman Institute of Technology (fictional), 2022 - 2026
CGPA: 8.7/10
Relevant coursework: Machine Learning, Data Structures, Probability and Statistics, DBMS

EXPERIENCE
Machine Learning Intern, Nimbus Analytics (fictional) | May 2025 - Aug 2025
- Built a text classification pipeline with scikit-learn and PyTorch that improved F1 from 0.71 to 0.84 on 40k support tickets.
- Fine-tuned a DistilBERT model using Hugging Face Transformers and served it with FastAPI and Docker.
- Automated data cleaning with Pandas and NumPy, reducing preprocessing time by 60%.
Research Assistant, University NLP Lab | Jan 2025 - Apr 2025
- Implemented named entity recognition experiments in spaCy and compared them with BERT baselines.
- Tracked 30+ experiments with MLflow and documented results for the lab.

PROJECTS
- Semantic Job Search: indexed 20,000 job posts with Sentence Transformers and FAISS; median query latency 45 ms.
- RAG Study Assistant: LangChain + Gemini API chatbot over lecture PDFs with source citations; used by 120 classmates.
- Crop Price Forecasting: XGBoost model with feature engineering on 5 years of mandi data; MAPE 8.2%.

SKILLS
Languages: Python, SQL, Java, C++
ML/AI: Machine Learning, Deep Learning, NLP, PyTorch, TensorFlow, scikit-learn, XGBoost, Transformers, Sentence Transformers
Data: Pandas, NumPy, Matplotlib, Jupyter, PostgreSQL
Tools: Git, GitHub Actions, Docker, FastAPI, Streamlit, Linux, AWS (S3, EC2)

ACHIEVEMENTS
- Winner, Smart India Hackathon 2024 internal round (team of 6).
- Published a workshop paper on low-resource Telugu text classification.
""",
"karthik_menon_fullstack": """Karthik Menon
karthik.menon@example.com | +91 55500 44556 | linkedin.com/in/karthik-menon-demo | github.com/karthik-menon-demo

PROFESSIONAL SUMMARY
Full-stack developer with 1.5 years of experience building React and Node.js applications for SaaS products.

EXPERIENCE
Software Engineer, BrightCart Commerce (fictional) | Jul 2024 - Present
- Developed React and TypeScript dashboards used by 3,000 merchants; cut page load time by 35%.
- Designed REST APIs with Node.js and Express backed by MongoDB and Redis.
- Set up CI/CD with GitHub Actions and Docker, reducing release time from 2 days to 3 hours.
Software Engineering Intern, PixelForge Studio (fictional) | Jan 2024 - Jun 2024
- Built reusable UI components in React with Tailwind CSS.
- Wrote Jest unit tests raising coverage from 40% to 78%.

PROJECTS
- Event Booking Platform: Next.js, Node.js, PostgreSQL, Stripe test mode; deployed on AWS EC2.
- Realtime Chat: Socket.io and React with JWT authentication.

EDUCATION
B.E. in Information Technology, Coastal Engineering College (fictional), 2020 - 2024, CGPA 8.1/10

SKILLS
JavaScript, TypeScript, React, Next.js, Node.js, Express.js, HTML, CSS, Tailwind, MongoDB, PostgreSQL, Redis,
Docker, AWS, Git, Jest, REST APIs, Agile
""",
"meera_iyer_data_analyst": """Meera Iyer
meera.iyer@example.com | +91 55500 77889 | linkedin.com/in/meera-iyer-demo

OBJECTIVE
Recent statistics graduate seeking a data analyst role where I can turn data into business decisions.

EDUCATION
B.Sc in Statistics, Lakeview College for Women (fictional), 2022 - 2025
CGPA: 8.9/10

INTERNSHIPS
Data Analyst Intern, FreshBasket Retail (fictional) | Jan 2025 - Jun 2025
- Analyzed 1.2M sales transactions in SQL and Excel to identify the top 20 slow-moving SKUs.
- Built a Power BI dashboard with DAX measures used weekly by 4 regional managers.
- Performed hypothesis testing on a pricing experiment; recommended a change that lifted margin by 3%.

PROJECTS
- Customer Segmentation: K-means clustering in Python (Pandas, scikit-learn) on 50k customers.
- COVID Data Story: Tableau Public dashboard with 10k+ views.

SKILLS
Excel (pivot tables, VLOOKUP), SQL, MySQL, Power BI, Tableau, Python, Pandas, Statistics, Data Visualization,
Data Cleaning, Communication
""",
"rohan_das_needs_work": """Rohan Das
rohan.das.demo@example.com

About me
I am a hardworking and passionate team player looking for a job in IT. I am a quick learner and I am very
detail-oriented.

Education
Bachelor of Computer Applications, City Degree College (fictional), 2021 - 2024

Work
Responsible for helping the team with website work at a local shop.
Worked on some Python scripts.
Helped in testing.

Skills
MS Office, Python, HTML, communication
""",
}

JOBS: dict[str, str] = {
"ai_ml_engineer_intern": """Job Title: AI/ML Engineer Intern
Company: Nimbus Labs (fictional sample)
Location: Remote, India

About the role
We are looking for an AI/ML Engineer Intern to build NLP features for our career-tech platform.

Responsibilities
- Build and evaluate machine learning models for text classification and semantic search.
- Develop NLP pipelines using spaCy and Hugging Face Transformers.
- Create embeddings with Sentence Transformers and integrate them with a vector database.
- Expose models through REST APIs using FastAPI and package them with Docker.
- Prototype LLM features with prompt engineering and RAG.

Requirements
- Pursuing or completed a Bachelor's degree in Computer Science or a related field.
- Strong Python, Pandas and NumPy skills.
- Solid understanding of machine learning fundamentals and model evaluation.
- Experience with PyTorch or TensorFlow.
- Familiarity with Git and SQL.

Nice to have
- Experience with LangChain, MLflow or AWS.
- Exposure to Streamlit for quick demos.
""",
"data_analyst_intern": """Job Title: Data Analyst Intern
Company: Riverstone Insights (fictional sample)

Responsibilities
- Clean, analyze and visualize business data to answer stakeholder questions.
- Build and maintain dashboards in Power BI or Tableau.
- Write SQL queries against MySQL and PostgreSQL databases.
- Present findings clearly to non-technical teams.

Requirements
- Bachelor's degree in Statistics, Mathematics, Economics, Computer Science or similar.
- Strong Excel skills including pivot tables.
- Good knowledge of SQL and statistics (hypothesis testing).
- Data visualization and communication skills.

Preferred
- Python with Pandas for data analysis.
- Exposure to A/B testing.
""",
"full_stack_developer": """Job Title: Full Stack Developer
Company: Orbit Commerce (fictional sample)

We are hiring a Full Stack Developer with 1+ years of experience to build merchant-facing web apps.

What you'll do
- Build responsive front-end features in React and TypeScript.
- Design and implement REST APIs with Node.js and Express.
- Work with MongoDB and PostgreSQL data models.
- Write unit tests and participate in code reviews.

Requirements
- 1+ years of experience in full-stack web development.
- Strong JavaScript, TypeScript, HTML and CSS.
- Experience with React and Node.js.
- Experience with Git and CI/CD pipelines.
- Bachelor's degree in Computer Science or equivalent experience.

Nice to have
- Docker, AWS and GraphQL.
- Experience with Redis caching.
""",
"nlp_engineer_junior": """Job Title: Junior NLP Engineer
Company: LinguaWorks AI (fictional sample)

Responsibilities
- Train and fine-tune transformer models for Indian-language text classification and NER.
- Build retrieval-augmented generation (RAG) systems with vector databases such as FAISS.
- Evaluate large language models and design prompts.
- Deploy models with Docker on AWS or GCP.

Requirements
- 1+ years of experience (internships count) in NLP or machine learning.
- Strong Python and PyTorch.
- Hands-on experience with Hugging Face Transformers, spaCy and Sentence Transformers.
- Understanding of deep learning and statistics.
- Bachelor's or Master's degree in Computer Science.

Preferred
- Experience with fine-tuning (LoRA/PEFT) and LangChain.
- Publications in NLP venues.
""",
"mlops_engineer": """Job Title: MLOps Engineer
Company: Stratus Cloud Labs (fictional sample)

Responsibilities
- Build CI/CD pipelines for training and deploying ML models.
- Operate Kubernetes clusters and containerized model services.
- Implement experiment tracking and model monitoring with MLflow.
- Automate infrastructure with Terraform on AWS.

Requirements
- 2+ years of experience in DevOps or MLOps.
- Strong Python and Linux skills, Bash scripting.
- Docker, Kubernetes and CI/CD (GitHub Actions or Jenkins).
- Cloud experience on AWS or GCP.
- Bachelor's degree in Computer Science or related field.

Nice to have
- Apache Airflow and Kafka.
- Knowledge of machine learning fundamentals.
""",
"backend_java_developer": """Job Title: Backend Java Developer
Company: Ledgerline Fintech (fictional sample)

Responsibilities
- Develop microservices in Java with Spring Boot.
- Design REST APIs and integrate with Kafka event streams.
- Optimize PostgreSQL queries and Redis caches.

Requirements
- 2+ years of experience with Java and Spring Boot.
- Strong understanding of OOP, data structures and algorithms.
- Experience with SQL, PostgreSQL and microservices.
- Familiarity with Docker and Kubernetes.
- Bachelor's degree in Computer Science.

Preferred
- Experience with AWS and system design.
""",
}


def _write_pdf(path: Path, text: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=12.5)
    head = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11, leading=15, spaceBefore=6)
    name = ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=16, leading=20)
    el = []
    for i, line in enumerate(text.strip().split("\n")):
        if not line.strip():
            el.append(Spacer(1, 4))
        elif i == 0:
            el.append(Paragraph(escape(line), name))
        elif line.isupper() and len(line) < 40:
            el.append(Paragraph(escape(line), head))
        elif line.startswith("- "):
            el.append(Paragraph("• " + escape(line[2:]), body))
        else:
            el.append(Paragraph(escape(line), body))
    SimpleDocTemplate(str(path), pagesize=A4, title=path.stem, author="Cynbit Studio sample").build(el)


def _write_docx(path: Path, text: str) -> None:
    import docx

    d = docx.Document()
    for i, line in enumerate(text.strip().split("\n")):
        if i == 0:
            d.add_heading(line, level=1)
        elif line.isupper() and len(line) < 40:
            d.add_heading(line.title(), level=2)
        elif line.startswith("- "):
            d.add_paragraph(line[2:], style="List Bullet")
        elif line.strip():
            d.add_paragraph(line)
    d.save(str(path))


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    _write_pdf(RES_DIR / "ananya_rao_ml_student.pdf", RESUMES["ananya_rao_ml_student"])
    _write_docx(RES_DIR / "karthik_menon_fullstack.docx", RESUMES["karthik_menon_fullstack"])
    _write_pdf(RES_DIR / "meera_iyer_data_analyst.pdf", RESUMES["meera_iyer_data_analyst"])
    (RES_DIR / "rohan_das_needs_work.txt").write_text(RESUMES["rohan_das_needs_work"], encoding="utf-8")
    _write_docx(RES_DIR / "rohan_das_needs_work.docx", RESUMES["rohan_das_needs_work"])
    for name, text in JOBS.items():
        (JOB_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    print(f"Wrote samples to {RES_DIR} and {JOB_DIR}")


if __name__ == "__main__":
    main()
