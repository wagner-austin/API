"""Outlook (Graph) HTTP plumbing: headers, verbs, error mapping."""

from __future__ import annotations

from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from platform_email.config import OUTLOOK_API_BASE
from platform_email.testing import HTTPErrorProtocol, hooks


class _OutlookHttp:
    """Microsoft Outlook email client using Graph API."""

    def __init__(self, *, access_token: str) -> None:
        """Initialize the client.

        Args:
            access_token: OAuth access token.
        """
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        """Get standard request headers.

        Returns:
            Headers dict with Authorization and Content-Type.
        """
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, status_code: int, message: str, context: str) -> None:
        """Handle HTTP error response.

        Args:
            status_code: HTTP status code.
            message: Error message.
            context: Context for error (URL path).

        Raises:
            AppError[EmailErrorCode]: Always raises.
        """
        if status_code == 404:
            if "messages" in context or "message" in context:
                msg = f"Email not found: {message}"
                raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
            if "folder" in context.lower():
                msg = f"Folder not found: {message}"
                raise AppError(EmailErrorCode.FOLDER_NOT_FOUND, msg, http_status=404)
            if "draft" in context.lower():
                msg = f"Draft not found: {message}"
                raise AppError(EmailErrorCode.DRAFT_NOT_FOUND, msg, http_status=404)
            msg = f"Resource not found: {message}"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
        msg = f"API error ({status_code}): {message}"
        raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=status_code)

    def _get(self, path: str) -> JSONObject:
        """Make a GET request to Graph API.

        Args:
            path: API path (appended to base URL).

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        try:
            response = hooks.http_get(url, self._headers())
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                body_text: str = e.read().decode("utf-8")
                self._handle_error(status_code, body_text, path)
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return data

    def _post(self, path: str, body: JSONObject) -> JSONObject:
        """Make a POST request to Graph API.

        Args:
            path: API path.
            body: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        body_str = dump_json_str(body)
        try:
            response = hooks.http_post(url, self._headers(), body_str)
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, path)
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return data

    def _patch(self, path: str, body: JSONObject) -> JSONObject:
        """Make a PATCH request to Graph API.

        Args:
            path: API path.
            body: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        body_str = dump_json_str(body)
        try:
            response = hooks.http_patch(url, self._headers(), body_str)
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, path)
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return data

    def _delete(self, path: str) -> None:
        """Make a DELETE request to Graph API.

        Args:
            path: API path.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        try:
            hooks.http_delete(url, self._headers())
        except ConnectionError as e:
            msg = f"Delete request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, path)
            msg = f"Delete request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
