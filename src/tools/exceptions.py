"""
Exception hierarchy for the agent tool layer.

Keeping these distinct (rather than raising bare Exception or ValueError
everywhere) lets the orchestrator distinguish between "the caller gave me
bad input" and "something broke while I was running" - which matters for
an agent loop deciding whether to retry, reformulate, or give up.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for all tool-related errors. Catch this to handle any
    failure originating from the tool layer without swallowing unrelated
    exceptions elsewhere in the application."""


class ToolInputError(ToolError):
    """Raised when a tool is constructed or invoked with invalid arguments.

    This is a caller/programmer error (bad config, bad call signature) as
    opposed to a runtime failure, and is typically not worth retrying.
    """


class ToolExecutionError(ToolError):
    """Raised when a tool fails while doing its job (I/O error, vector
    store failure, etc.) despite having received valid input.

    An agent orchestrator may choose to retry, fall back to another tool,
    or surface this to the user, depending on context.
    """


class PathSecurityError(ToolError):
    """Raised when a requested file path would resolve outside the
    repository root that a filesystem-facing tool is scoped to.

    Always raised instead of silently clamping or ignoring the path -
    silent path correction is how directory traversal bugs slip through.
    """
