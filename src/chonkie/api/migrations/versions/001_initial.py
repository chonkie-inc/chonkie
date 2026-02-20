"""Initial migration: create pipelines table.

Revision ID: 001
Revises:
Create Date: 2026-02-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the pipelines table."""
    op.create_table(
        "pipelines",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_pipelines_name"), "pipelines", ["name"], unique=True)


def downgrade() -> None:
    """Drop the pipelines table."""
    op.drop_index(op.f("ix_pipelines_name"), table_name="pipelines")
    op.drop_table("pipelines")
