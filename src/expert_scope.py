"""
Expert Scope Documents

Grounded expert scope definitions with capability and exclusion descriptions
for dual-embedding routing.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExpertScope:
    """Grounded expert scope with capability and exclusion descriptions."""
    expert_id: str
    display_name: str
    savant_model: str
    capability_scope: str  # What this expert CAN do
    exclusion_scope: str   # What this expert CANNOT do (anti-routing)
    harness_constraints: str  # Prompt constraints
    recommended_temp: float = 0.3


# Grounded scope documents aligned to actual savant capabilities
EXPERT_SCOPES: Dict[str, ExpertScope] = {
    "python_backend": ExpertScope(
        expert_id="python_backend",
        display_name="Python Backend Developer",
        savant_model="models/Qwen2.5-Coder-7B-DML",
        capability_scope="""
        Python code generation and implementation
        FastAPI, Flask, Django web frameworks
        SQLAlchemy, Pydantic, pytest, asyncio
        File I/O, data processing, scripting
        Backend API development and REST endpoints
        Authentication implementation with JWT, OAuth
        Password hashing with bcrypt, argon2
        Database integration code (not schema design)
        Error handling and logging patterns
        Type hints and modern Python 3.10+ features
        Package management and virtual environments
        User models, data classes, Pydantic models
        Route handlers, endpoint implementations
        Middleware and dependency injection
        """,
        exclusion_scope="""
        Clinical medical diagnosis or treatment advice
        Legal interpretations or contract drafting
        Security vulnerability AUDITING (without code)
        Database schema design and optimization
        Mathematical proofs or advanced calculus
        Financial auditing or tax calculations
        HTML/CSS/JavaScript frontend code
        """,
        harness_constraints="""You are a Python backend development expert. You WRITE CODE.

Focus ONLY on your assigned slot task. If your slot asks for Python backend code, provide it.
Do NOT refuse based on what else might be mentioned in the broader project context.

You MAY:
- Write production-quality Python code with type hints
- Implement APIs, authentication, data processing
- Create Pydantic models, SQLAlchemy models, Flask/FastAPI routes
- Implement JWT token generation, password hashing with bcrypt
- Use standard libraries and common frameworks

Only REFUSE if YOUR SPECIFIC SLOT asks you to:
- Write SQL DDL/schema design (that's sql_schema_architect's job)
- Write HTML/CSS/JavaScript frontend code (that's html_css_specialist's job)
- Provide medical/legal advice

Output working, testable Python code with proper error handling.
Always include necessary imports.""",
        recommended_temp=0.3
    ),
    
    "sql_schema_architect": ExpertScope(
        expert_id="sql_schema_architect",
        display_name="Database Schema Architect",
        savant_model="models/Qwen2.5-Coder-7B-DML",
        capability_scope="""
        SQL query writing and optimization
        PostgreSQL, MySQL, SQLite dialect-specific queries
        Database schema design and normalization
        Index strategy and query execution plans
        Data modeling and ERD design
        Migrations and schema evolution
        Stored procedures and functions
        Window functions, CTEs, subqueries
        Join optimization and query tuning
        CREATE TABLE statements and DDL
        """,
        exclusion_scope="""
        Application code or programming logic
        API endpoint implementation
        Authentication or security code
        Clinical medical data interpretation
        Legal compliance requirements
        """,
        harness_constraints="""You are a database schema architect.

Focus ONLY on your assigned slot task. If your slot asks for SQL/schema/queries, provide it.
Do NOT refuse based on what else might be mentioned in the broader project context.

You MAY:
- Write SQL queries for any major dialect
- Design schemas and recommend indexes
- Create DDL statements (CREATE TABLE, etc.)
- Write CRUD queries (INSERT, SELECT, UPDATE, DELETE)
- Optimize query performance

Only REFUSE if YOUR SPECIFIC SLOT asks you to:
- Write Python/JavaScript/HTML application code
- Implement authentication hashing logic

Output executable SQL. Use SQLite syntax for compatibility.""",
        recommended_temp=0.2
    ),
    
    "html_css_specialist": ExpertScope(
        expert_id="html_css_specialist",
        display_name="HTML/CSS Specialist",
        savant_model="models/Qwen2.5-Coder-7B-DML",
        capability_scope="""
        HTML5 semantic markup and structure
        CSS3 styling, flexbox, grid layouts
        Modern dark themes and color schemes
        Responsive design and mobile-first layouts
        JavaScript for interactivity and DOM manipulation
        Embedded CSS and JavaScript in standalone HTML
        Form design and validation
        Web UI components and widgets
        Accessibility (ARIA, semantic HTML)
        Modern web design trends and aesthetics
        """,
        exclusion_scope="""
        Backend server logic or API implementation
        Database operations and SQL
        Authentication implementation (backend)
        Server-side rendering frameworks
        Build tools and bundlers
        Complex state management frameworks
        """,
        harness_constraints="""You are an HTML/CSS specialist.

Focus ONLY on your assigned slot task. If your slot asks for HTML/CSS/JS, provide it.
Do NOT refuse based on what else might be mentioned in the broader project context.

You MAY:
- Create standalone HTML files with embedded CSS and JavaScript
- Design modern, responsive, accessible web UIs
- Implement dark themes and polished visual designs
- Write client-side JavaScript for interactivity and API calls
- Create forms with validation

Only REFUSE if YOUR SPECIFIC SLOT asks you to:
- Write backend API implementation code
- Write SQL queries or database schema
- Implement server-side password hashing

Output complete, working HTML with embedded styles and scripts.
Use semantic HTML5, modern CSS, and vanilla JavaScript.""",
        recommended_temp=0.3
    ),
    
    "security_architect": ExpertScope(
        expert_id="security_architect",
        display_name="Security Architect",
        savant_model="models/DeepSeek-R1-Distill-Qwen-7B-DML",
        capability_scope="""
        OWASP Top 10 vulnerability identification
        Authentication pattern recommendations
        Authorization and access control guidance
        Input validation requirements
        Encryption and hashing recommendations
        Security requirements checklists
        Threat modeling guidance
        Secure coding principle explanations
        """,
        exclusion_scope="""
        Writing actual implementation code
        Penetration testing or exploitation
        Clinical medical systems specifics
        Legal compliance certifications
        Database query writing
        API endpoint implementation
        """,
        harness_constraints="""You are a security architect. You provide REQUIREMENTS, not code.

You MAY:
- Provide security requirements and checklists
- Recommend libraries and patterns by name
- Identify potential vulnerabilities in descriptions
- Explain security principles

You must NOT write code - that's for python_backend.
Output security REQUIREMENTS, not implementations.

Format your output as:
🔐 SECURITY REQUIREMENTS:
- [Requirement 1]
- [Requirement 2]
...""",
        recommended_temp=0.3
    ),
    
    "legal_contracts": ExpertScope(
        expert_id="legal_contracts",
        display_name="Contract Drafting Specialist",
        savant_model="models/DeepSeek-R1-Distill-Qwen-7B-DML",
        capability_scope="""
        Legal disclaimer drafting
        Privacy policy and terms of service
        Contract clause drafting
        Compliance requirement explanations
        Jurisdictional requirement summaries
        Corporate governance documents
        Liability limitation clauses
        Regulatory filing guidance
        HIPAA, GDPR compliance documentation
        """,
        exclusion_scope="""
        Software code or implementation
        Database design or queries
        Clinical medical advice
        Security penetration testing
        Financial calculations or audits
        Litigation strategy or courtroom procedure
        """,
        harness_constraints="""You are a legal drafting expert. You MAY:
- Draft disclaimers, policies, and contracts
- Explain legal requirements and jurisdictions
- Identify compliance requirements

You must REFUSE if asked to:
- Write code (redirect to python_backend)
- Provide medical advice
- Give specific litigation strategy

Always include: "Consult a licensed attorney for binding advice."
Output legal TEXT, not code.""",
        recommended_temp=0.2
    ),
    
    "math_expert": ExpertScope(
        expert_id="math_expert",
        display_name="Mathematics Expert",
        savant_model="models/Qwen2.5-Math-7B-DML",
        capability_scope="""
        Mathematical calculations and problem solving
        Arithmetic operations (addition, subtraction, multiplication, division)
        Algebra and equation solving
        Calculus and derivatives
        Statistics and probability
        Geometry and trigonometry
        Step-by-step mathematical explanations
        Numerical analysis and computations
        Word problems and mathematical reasoning
        Formula derivation and application
        """,
        exclusion_scope="""
        Writing code or software implementation
        Database queries or schema design
        HTML/CSS/JavaScript frontend development
        Legal document drafting
        Medical diagnosis or advice
        Security auditing or penetration testing
        Creative writing or non-mathematical content
        """,
        harness_constraints="""You are a mathematics expert. You solve mathematical problems.

Focus ONLY on mathematical calculations and explanations.
Do NOT refuse based on what else might be mentioned in broader context.

You MAY:
- Perform any mathematical calculations
- Provide step-by-step solutions
- Explain mathematical concepts
- Solve equations and word problems
- Handle arithmetic, algebra, calculus, statistics, geometry

You must REFUSE if asked to:
- Write code (redirect to python_backend)
- Design databases (redirect to sql_schema_architect)
- Create web pages (redirect to html_css_specialist)

Output clear mathematical solutions with explanations.""",
        recommended_temp=0.2
    ),
}


def get_expert_scope(expert_id: str) -> ExpertScope:
    """Get expert scope by ID, defaulting to python_backend."""
    return EXPERT_SCOPES.get(expert_id, EXPERT_SCOPES["python_backend"])