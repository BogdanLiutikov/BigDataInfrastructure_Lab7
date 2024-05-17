from datetime import datetime
from typing import List

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Predict(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(primary_key=True)  # autoincrement=True
    x: Mapped[str]
    # y_true: Mapped[float | None]
    y_pred: Mapped[float | None]
    datatime: Mapped[datetime] = mapped_column(server_default=func.now())
