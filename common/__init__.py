"""Shared foundation for the portfolio projects: LLM access, tracing, config and statistics.

Nothing in here knows about research analysts, agents or SQL. Project packages import from
`common`; `common` never imports from them.
"""
