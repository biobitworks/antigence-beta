"""
Database models and initialization for IMMUNOS-MCP web application
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Pattern(Base):
    """Training patterns table"""
    __tablename__ = 'patterns'

    id = Column(Integer, primary_key=True)
    domain = Column(String(50), nullable=False)  # 'security', 'emotion', 'spam', 'network'
    label = Column(String(100), nullable=False)
    data = Column(Text, nullable=False)
    embedding = Column(LargeBinary)  # NumPy embedding stored as bytes
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class Analysis(Base):
    """Analysis results table"""
    __tablename__ = 'analyses'

    id = Column(Integer, primary_key=True)
    domain = Column(String(50), nullable=False)
    input_data = Column(Text, nullable=False)
    predicted_label = Column(String(100))
    confidence = Column(Float)
    agent_used = Column(String(100))  # 'bcell', 'nkcell', 'ensemble'
    execution_time = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class UserSubmission(Base):
    """User training submissions table"""
    __tablename__ = 'user_submissions'

    id = Column(Integer, primary_key=True)
    domain = Column(String(50), nullable=False)
    data = Column(Text, nullable=False)
    user_label = Column(String(100), nullable=False)
    accepted = Column(Boolean, default=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)


class Metric(Base):
    """System metrics table"""
    __tablename__ = 'metrics'

    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class OpsEvent(Base):
    """Operational events (detections, sentinel ingest, etc.)"""

    __tablename__ = "ops_events"

    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)  # detection|sentinel|...
    source = Column(String(100), nullable=False)  # immunos|sentinel|...
    severity = Column(String(20), nullable=True)  # low|med|high
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Ticket(Base):
    """High-risk items that require review/fix."""

    __tablename__ = "tickets"
    __table_args__ = (UniqueConstraint("fingerprint", name="uq_ticket_fingerprint"),)

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="open")  # open|closed
    severity = Column(String(20), nullable=False)  # low|med|high
    source = Column(String(100), nullable=False)  # sentinel|immunos|manual

    source_event_ts = Column(String(64), nullable=True)
    fingerprint = Column(String(64), nullable=False)

    summary = Column(Text, nullable=False)
    details_json = Column(Text, nullable=False)

    resolved_at = Column(DateTime, nullable=True)
    resolution_note = Column(Text, nullable=True)


def init_database(db_path='data/immunos.db'):
    """
    Initialize database with schema

    Args:
        db_path: Path to SQLite database file
    """
    # Create data directory if it doesn't exist
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(f'sqlite:///{db_path}')

    # Create all tables
    Base.metadata.create_all(engine)

    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f"✅ Database initialized at: {db_path}")
    print("✅ Tables created: patterns, analyses, user_submissions, metrics, ops_events, tickets")

    return engine, session


def seed_database(session):
    """
    Seed database with initial sample data

    Args:
        session: SQLAlchemy session
    """
    # Sample security patterns
    security_patterns = [
        Pattern(
            domain='security',
            label='safe',
            data='def validate_input(user_input):\n    return user_input.strip()',
            confidence=0.95
        ),
        Pattern(
            domain='security',
            label='vulnerable',
            data='def execute_command(cmd):\n    os.system(cmd)  # Command injection!',
            confidence=0.98
        ),
    ]

    # Sample emotion patterns
    emotion_patterns = [
        Pattern(
            domain='emotion',
            label='joy',
            data='I am so happy today! This is wonderful!',
            confidence=0.92
        ),
        Pattern(
            domain='emotion',
            label='sadness',
            data='I feel really down and depressed.',
            confidence=0.89
        ),
    ]

    # Sample spam patterns
    spam_patterns = [
        Pattern(
            domain='spam',
            label='spam',
            data='CONGRATULATIONS! You won $1,000,000! Click here NOW!',
            confidence=0.97
        ),
        Pattern(
            domain='spam',
            label='ham',
            data='Hi John, let me know when you are free for our meeting tomorrow.',
            confidence=0.94
        ),
    ]

    # Add all patterns
    all_patterns = security_patterns + emotion_patterns + spam_patterns
    for pattern in all_patterns:
        session.add(pattern)

    # Sample metrics
    metrics = [
        Metric(metric_name='total_patterns', value=len(all_patterns)),
        Metric(metric_name='total_analyses', value=0),
        Metric(metric_name='avg_confidence', value=0.94),
    ]

    for metric in metrics:
        session.add(metric)

    session.commit()
    print(f"✅ Seeded {len(all_patterns)} sample patterns")
    print(f"✅ Seeded {len(metrics)} metrics")


def get_stats(session):
    """
    Get database statistics

    Args:
        session: SQLAlchemy session

    Returns:
        dict: Database statistics
    """
    stats = {
        'total_patterns': session.query(Pattern).count(),
        'total_analyses': session.query(Analysis).count(),
        'total_submissions': session.query(UserSubmission).count(),
        'total_tickets': session.query(Ticket).count(),
        'open_tickets': session.query(Ticket).filter_by(status='open').count(),
        'patterns_by_domain': {},
        'analyses_by_domain': {},
    }

    # Count patterns by domain
    for domain in ['security', 'emotion', 'spam', 'network', 'hallucination']:
        stats['patterns_by_domain'][domain] = session.query(Pattern).filter_by(domain=domain).count()
        stats['analyses_by_domain'][domain] = session.query(Analysis).filter_by(domain=domain).count()

    return stats


def create_ticket(
    session,
    *,
    severity: str,
    source: str,
    summary: str,
    details: dict,
    fingerprint: str,
    source_event_ts: str = None,
):
    """Idempotent create: returns existing ticket if fingerprint exists."""
    existing = session.query(Ticket).filter_by(fingerprint=fingerprint).first()
    if existing:
        return existing

    ticket = Ticket(
        severity=severity,
        source=source,
        summary=summary,
        details_json=json.dumps(details, ensure_ascii=False, sort_keys=True),
        fingerprint=fingerprint,
        source_event_ts=source_event_ts,
    )
    session.add(ticket)
    session.flush()
    return ticket


def close_ticket(session, ticket_id: int, note: str = ""):
    ticket = session.query(Ticket).filter_by(id=ticket_id).first()
    if not ticket:
        return None
    ticket.status = "closed"
    ticket.resolved_at = datetime.utcnow()
    ticket.resolution_note = note
    session.add(ticket)
    session.flush()
    return ticket


if __name__ == '__main__':
    """Command-line interface for database management"""

    if len(sys.argv) < 2:
        print("Usage: python database.py [init|seed|stats]")
        sys.exit(1)

    command = sys.argv[1]
    db_path = 'data/immunos.db' if len(sys.argv) < 3 else sys.argv[2]

    if command == 'init':
        engine, session = init_database(db_path)
        print("\n✅ Database ready for use")

    elif command == 'seed':
        engine, session = init_database(db_path)
        seed_database(session)
        print("\n✅ Database seeded with sample data")

    elif command == 'stats':
        engine = create_engine(f'sqlite:///{db_path}')
        Session = sessionmaker(bind=engine)
        session = Session()
        stats = get_stats(session)

        print("\n📊 Database Statistics:")
        print(f"   Total patterns: {stats['total_patterns']}")
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Total submissions: {stats['total_submissions']}")
        print("\n   Patterns by domain:")
        for domain, count in stats['patterns_by_domain'].items():
            print(f"     {domain}: {count}")

    else:
        print(f"❌ Unknown command: {command}")
        print("Usage: python database.py [init|seed|stats]")
        sys.exit(1)
