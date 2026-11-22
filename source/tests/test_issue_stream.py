"""Tests for IssueStream behavior."""

from __future__ import annotations

import typing as t

import pytest

from tap_jira.streams import (
    IssueChangeLogStream,
    IssueComments,
    IssueStream,
    IssueWatchersStream,
    IssueWorklogs,
)
from tap_jira.tap import TapJira


def make_tap(**config_overrides: object) -> TapJira:
    config: dict[str, object] = {
        "domain": "test.atlassian.net",
        "email": "test@example.com",
        "api_token": "test-token",
    }
    config.update(config_overrides)
    return TapJira(config=config)


def make_issue_record(
    issue_id: str,
    key: str,
    project_key: str,
    updated: str,
) -> dict[str, t.Any]:
    """Build a raw issue search record as returned by the Jira API."""
    return {
        "id": issue_id,
        "key": key,
        "self": f"https://test.atlassian.net/rest/api/3/issue/{issue_id}",
        "expand": "",
        "fields": {
            "project": {
                "self": "https://test.atlassian.net/rest/api/3/project/1",
                "id": "1",
                "key": project_key,
                "name": f"Project {project_key}",
                "projectTypeKey": "software",
            },
            "summary": f"Summary for {key}",
            "created": "2026-01-01T00:00:00.000+0000",
            "updated": updated,
        },
    }


def patch_records(
    monkeypatch: pytest.MonkeyPatch,
    stream: IssueStream,
    records: list[dict[str, t.Any]],
) -> None:
    """Make the stream yield the given records instead of calling the API."""
    monkeypatch.setattr(stream, "request_records", lambda _context: iter(records))


def test_get_url_params_jql_start_date() -> None:
    tap = TapJira(
        config={
            "domain": "test.atlassian.net",
            "email": "test@example.com",
            "api_token": "test-token",
        },
        state={
            "bookmarks": {
                "issues": {
                    "starting_replication_value": "2026-01-01T12:34:56",
                },
            },
        },
    )

    stream = IssueStream(tap)
    params = stream.get_url_params(context=None, next_page_token=None)

    assert params["jql"] == (
        "(created>='2026-01-01 12:34' or updated>='2026-01-01 12:34') and (id != null) "
        "order by updated asc"
    )


def test_get_url_params_jql_start_and_end_date() -> None:
    tap = TapJira(
        config={
            "domain": "test.atlassian.net",
            "email": "test@example.com",
            "api_token": "test-token",
            "end_date": "2026-02-01T00:00:00",
        },
        state={
            "bookmarks": {
                "issues": {
                    "starting_replication_value": "2026-01-01T12:34:56",
                },
            },
        },
    )

    stream = IssueStream(tap)
    params = stream.get_url_params(context=None, next_page_token=None)

    assert params["jql"] == (
        "(created>='2026-01-01 12:34' or updated>='2026-01-01 12:34') and "
        "(created<'2026-02-01 00:00' or updated<'2026-02-01 00:00') and (id != null) "
        "order by updated asc"
    )


class TestIssueProjectMoves:
    """An issue moved between projects keeps its id, key history aside."""

    def test_moved_issue_is_emitted_once_with_latest_project(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same issue id surfaced under two projects yields one record."""
        tap = make_tap()
        stream = IssueStream(tap)
        before_move = make_issue_record(
            "10001",
            key="OLD-1",
            project_key="OLD",
            updated="2026-01-02T10:00:00.000+0000",
        )
        after_move = make_issue_record(
            "10001",
            key="NEW-1",
            project_key="NEW",
            updated="2026-01-03T10:00:00.000+0000",
        )
        other = make_issue_record(
            "10002",
            key="OLD-2",
            project_key="OLD",
            updated="2026-01-04T10:00:00.000+0000",
        )
        patch_records(monkeypatch, stream, [before_move, after_move, other])

        records = list(stream.get_records(context=None))

        assert [record["id"] for record in records] == ["10001", "10002"]
        moved = records[0]
        assert moved["key"] == "NEW-1"
        assert moved["fields"]["project"]["key"] == "NEW"

    def test_stale_duplicate_does_not_replace_newer_version(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stale record arriving after the fresh one must not win."""
        tap = make_tap()
        stream = IssueStream(tap)
        after_move = make_issue_record(
            "10001",
            key="NEW-1",
            project_key="NEW",
            updated="2026-01-03T10:00:00.000+0000",
        )
        before_move = make_issue_record(
            "10001",
            key="OLD-1",
            project_key="OLD",
            updated="2026-01-02T10:00:00.000+0000",
        )
        patch_records(monkeypatch, stream, [after_move, before_move])

        records = list(stream.get_records(context=None))

        assert [record["key"] for record in records] == ["NEW-1"]

    def test_records_without_duplicates_pass_through_sorted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tap = make_tap()
        stream = IssueStream(tap)
        first = make_issue_record(
            "10001",
            key="PROJ-1",
            project_key="PROJ",
            updated="2026-01-02T10:00:00.000+0000",
        )
        second = make_issue_record(
            "10002",
            key="PROJ-2",
            project_key="PROJ",
            updated="2026-01-03T10:00:00.000+0000",
        )
        patch_records(monkeypatch, stream, [first, second])

        records = list(stream.get_records(context=None))

        assert [record["id"] for record in records] == ["10001", "10002"]

    def test_sync_publishes_single_active_record_for_moved_issue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end: one RECORD per issue id and state tracks the last one."""
        tap = make_tap()
        stream = IssueStream(tap)
        before_move = make_issue_record(
            "10001",
            key="OLD-1",
            project_key="OLD",
            updated="2026-01-02T10:00:00.000+0000",
        )
        after_move = make_issue_record(
            "10001",
            key="NEW-1",
            project_key="NEW",
            updated="2026-01-03T10:00:00.000+0000",
        )
        other = make_issue_record(
            "10002",
            key="OLD-2",
            project_key="OLD",
            updated="2026-01-04T10:00:00.000+0000",
        )
        patch_records(monkeypatch, stream, [before_move, after_move, other])
        monkeypatch.setattr(stream, "_sync_children", lambda _context: None)

        records = list(
            stream._sync_records(  # noqa: SLF001
                context=None,
                write_messages=False,
            ),
        )

        assert [record["id"] for record in records] == ["10001", "10002"]
        moved = records[0]
        assert moved["key"] == "NEW-1"
        assert moved["fields"]["project"]["key"] == "NEW"
        # post_process lifts created/updated to the top level
        assert moved["updated"] == "2026-01-03T10:00:00.000+0000"
        # The bookmark advances to the last emitted record, so the next
        # incremental sync resumes after the move instead of replaying it.
        assert stream.stream_state["replication_key_value"] == (
            "2026-01-04T10:00:00.000+0000"
        )


class TestIssueChildStreams:
    """Child records stay connected to the issue's immutable id."""

    def test_child_context_uses_real_issue_id(self) -> None:
        tap = make_tap()
        stream = IssueStream(tap)

        context = stream.get_child_context(
            {"id": "10001", "key": "NEW-1"},
            context=None,
        )

        assert context == {"issue_id": "10001"}

    @pytest.mark.parametrize(
        "stream_type",
        [IssueChangeLogStream, IssueComments, IssueWatchersStream, IssueWorklogs],
    )
    def test_child_records_are_stamped_with_real_issue_id(
        self,
        stream_type: type,
    ) -> None:
        tap = make_tap()
        stream = stream_type(tap)

        row = stream.post_process({"id": "1"}, {"issue_id": "10001"})

        assert row["issueId"] == "10001"

    def test_watchers_primary_keys_include_issue_id(self) -> None:
        """A watcher can watch many issues, so key by issue and account."""
        assert IssueWatchersStream.primary_keys == ("issueId", "accountId")
