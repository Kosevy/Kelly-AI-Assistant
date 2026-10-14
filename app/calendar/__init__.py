"""
app/calendar — Sistema de calendario local de Kelly.
"""
from app.calendar.exceptions import CalendarError, CalendarValidationError
from app.calendar.expander import (
    ExpandedBlock,
    expand_all,
    expand_events,
    expand_recurring_blocks,
)
from app.calendar.models import (
    BlockType,
    Calendar,
    Course,
    DayOfWeek,
    Event,
    EventType,
    Metadata,
    RecurringBlock,
)
from app.calendar.parser import load_calendar, save_calendar, validate_calendar
from app.calendar.query import events_in_range, free_slots, next_deadline

__all__ = [
    "BlockType",
    "Calendar",
    "CalendarError",
    "CalendarValidationError",
    "Course",
    "DayOfWeek",
    "Event",
    "EventType",
    "ExpandedBlock",
    "Metadata",
    "RecurringBlock",
    "events_in_range",
    "expand_all",
    "expand_events",
    "expand_recurring_blocks",
    "free_slots",
    "load_calendar",
    "next_deadline",
    "save_calendar",
    "validate_calendar",
]
