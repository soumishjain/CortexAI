
"""add message sequence cursor

Revision ID: f438da654e27
Revises: c813ba3cb669
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f438da654e27"
down_revision: Union[str, Sequence[str], None] = "c813ba3cb669"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add new columns while allowing existing rows to be backfilled.
    op.add_column(
        "conversations",
        sa.Column("summary_through_sequence", sa.Integer(), nullable=True),
    )

    op.add_column(
        "conversations",
        sa.Column(
            "next_message_sequence",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
    )

    op.add_column(
        "messages",
        sa.Column("sequence", sa.Integer(), nullable=True),
    )

    # 2. Assign deterministic sequence numbers to existing messages.
    # Historical ordering uses created_at, then UUID as a tie-breaker.
    op.execute(
        """
        WITH ranked_messages AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY created_at, id
                ) AS seq
            FROM messages
        )
        UPDATE messages AS m
        SET sequence = ranked_messages.seq
        FROM ranked_messages
        WHERE m.id = ranked_messages.id
        """
    )

    # 3. Translate the old UUID summary cursor into the new sequence cursor.
    op.execute(
        """
        UPDATE conversations AS c
        SET summary_through_sequence = m.sequence
        FROM messages AS m
        WHERE c.summary_through_message_id = m.id
          AND c.id = m.conversation_id
        """
    )

    # 4. Set the next available sequence for every existing conversation.
    op.execute(
        """
        UPDATE conversations AS c
        SET next_message_sequence = COALESCE(
            (
                SELECT MAX(m.sequence) + 1
                FROM messages AS m
                WHERE m.conversation_id = c.id
            ),
            1
        )
        """
    )

    # 5. Enforce required sequence values and uniqueness.
    op.alter_column(
        "messages",
        "sequence",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.create_unique_constraint(
        "uq_messages_conversation_sequence",
        "messages",
        ["conversation_id", "sequence"],
    )

    # 6. Remove the old UUID cursor after migrating its values.
    op.drop_column("conversations", "summary_through_message_id")


def downgrade() -> None:
    # Restore the old cursor before removing the sequence-based cursor.
    op.add_column(
        "conversations",
        sa.Column(
            "summary_through_message_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    op.execute(
        """
        UPDATE conversations AS c
        SET summary_through_message_id = m.id
        FROM messages AS m
        WHERE c.summary_through_sequence = m.sequence
          AND c.id = m.conversation_id
        """
    )

    op.drop_constraint(
        "uq_messages_conversation_sequence",
        "messages",
        type_="unique",
    )

    op.drop_column("messages", "sequence")
    op.drop_column("conversations", "next_message_sequence")
    op.drop_column("conversations", "summary_through_sequence")
