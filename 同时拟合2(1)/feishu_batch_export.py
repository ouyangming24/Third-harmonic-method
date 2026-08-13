#!/usr/bin/env python3
"""
飞书云文档批量导出脚本

支持：
  - 云空间文件夹（递归子文件夹）
  - 知识库 / 我的文档库（wiki）

用法示例：
  # 1. 先获取 user_access_token（个人文档必须用用户 token）
  #    npx -y @larksuiteoapi/lark-mcp login -a <app_id> -s <app_secret>

  # 2. 导出云空间某个文件夹为 docx
  python feishu_batch_export.py \\
    --token u-xxxx \\
    --source drive \\
    --folder-token fldxxxx \\
    --output ./exports \\
    --format docx

  # 3. 导出整个知识库为 pdf
  python feishu_batch_export.py \\
    --token u-xxxx \\
    --source wiki \\
    --space-id my_library \\
    --output ./exports \\
    --format pdf

所需飞书应用权限（在开放平台开通）：
  - 查看、评论和下载云空间中所有文件 (drive:drive:readonly)
  - 导出云文档 (drive:export:readonly 或 docs:document:export)
  - 查看知识库 (wiki:wiki:readonly)  # 仅 wiki 模式
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://open.feishu.cn/open-apis"

# 可导出的文档类型 -> (飞书 type, 允许的扩展名)
EXPORTABLE = {
    "docx": ("docx", ("docx", "pdf")),
    "doc": ("doc", ("docx", "pdf")),
    "sheet": ("sheet", ("xlsx", "csv")),
    "bitable": ("bitable", ("xlsx", "csv")),
}

POLL_INTERVAL = 2.0
POLL_MAX_WAIT = 120.0


def api_request(
    method: str,
    path: str,
    token: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    raw: bool = False,
) -> Any:
    url = f"{BASE}{path}"
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
        data = json.dumps(body).encode("utf-8")

    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            if raw:
                return resp.read()
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {path}: {detail}") from e

    if not raw and payload.get("code", -1) != 0:
        raise RuntimeError(f"API error {path}: {payload.get('msg')} (code={payload.get('code')})")
    return payload if not raw else payload


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    return name or "untitled"


def export_one(
    token: str,
    doc_token: str,
    doc_type: str,
    file_ext: str,
    dest: Path,
) -> None:
    if doc_type not in EXPORTABLE:
        print(f"  [跳过] 不支持导出类型: {doc_type}")
        return

    feishu_type, allowed_ext = EXPORTABLE[doc_type]
    if file_ext not in allowed_ext:
        print(f"  [跳过] {doc_type} 不支持导出为 .{file_ext}，可选: {allowed_ext}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [已存在] {dest}")
        return

    create_resp = api_request(
        "POST",
        "/drive/v1/export_tasks",
        token,
        body={
            "token": doc_token,
            "type": feishu_type,
            "file_extension": file_ext,
        },
    )
    ticket = create_resp["data"]["ticket"]

    file_token = None
    deadline = time.time() + POLL_MAX_WAIT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        result = api_request(
            "GET",
            f"/drive/v1/export_tasks/{ticket}",
            token,
            params={"token": doc_token},
        )
        job = result["data"]["result"]
        status = job["job_status"]
        if status == 0:
            file_token = job["file_token"]
            break
        if status in (1, 2):
            continue
        raise RuntimeError(f"导出失败 (job_status={status}): {job.get('job_error_msg', job)}")
    else:
        raise RuntimeError(f"导出超时: {doc_token}")

    content = api_request(
        "GET",
        f"/drive/v1/export_tasks/file/{file_token}/download",
        token,
        raw=True,
    )
    dest.write_bytes(content)
    print(f"  [完成] {dest}")


def list_drive_files(token: str, folder_token: str | None) -> list[dict]:
    files: list[dict] = []
    page_token = None
    while True:
        params: dict[str, Any] = {"page_size": 200}
        if folder_token:
            params["folder_token"] = folder_token
        if page_token:
            params["page_token"] = page_token

        resp = api_request("GET", "/drive/v1/files", token, params=params)
        files.extend(resp["data"].get("files", []))
        if not resp["data"].get("has_more"):
            break
        page_token = resp["data"].get("page_token")
    return files


def walk_drive(
    token: str,
    folder_token: str | None,
    rel_path: Path,
    file_ext: str,
    output_root: Path,
    stats: dict[str, int],
) -> None:
    items = list_drive_files(token, folder_token)
    for item in items:
        name = sanitize_filename(item["name"])
        item_type = item["type"]
        item_token = item["token"]

        if item_type == "folder":
            walk_drive(token, item_token, rel_path / name, file_ext, output_root, stats)
            continue

        if item_type == "shortcut":
            item_type = item.get("shortcut_info", {}).get("target_type", item_type)
            item_token = item.get("shortcut_info", {}).get("target_token", item_token)

        if item_type not in EXPORTABLE:
            stats["skipped"] += 1
            continue

        stats["total"] += 1
        dest = output_root / rel_path / f"{name}.{file_ext}"
        print(f"[{stats['total']}] {rel_path / name}")
        try:
            export_one(token, item_token, item_type, file_ext, dest)
            stats["ok"] += 1
        except Exception as e:
            stats["fail"] += 1
            print(f"  [失败] {e}")


def list_wiki_nodes(
    token: str,
    space_id: str,
    parent_node_token: str | None = None,
) -> list[dict]:
    nodes: list[dict] = []
    page_token = None
    while True:
        params: dict[str, Any] = {"page_size": 50}
        if parent_node_token:
            params["parent_node_token"] = parent_node_token
        if page_token:
            params["page_token"] = page_token

        resp = api_request(
            "GET",
            f"/wiki/v2/spaces/{space_id}/nodes",
            token,
            params=params,
        )
        nodes.extend(resp["data"].get("items", []))
        if not resp["data"].get("has_more"):
            break
        page_token = resp["data"].get("page_token")
    return nodes


def walk_wiki(
    token: str,
    space_id: str,
    parent_node_token: str | None,
    rel_path: Path,
    file_ext: str,
    output_root: Path,
    stats: dict[str, int],
) -> None:
    nodes = list_wiki_nodes(token, space_id, parent_node_token)
    for node in nodes:
        title = sanitize_filename(node.get("title") or "untitled")
        obj_type = node.get("obj_type", "")
        obj_token = node.get("obj_token", "")
        node_token = node.get("node_token")
        has_child = node.get("has_child", False)

        current_path = rel_path / title

        if obj_type and obj_token and obj_type in EXPORTABLE:
            stats["total"] += 1
            dest = output_root / current_path.parent / f"{title}.{file_ext}"
            print(f"[{stats['total']}] {current_path.parent / title}")
            try:
                export_one(token, obj_token, obj_type, file_ext, dest)
                stats["ok"] += 1
            except Exception as e:
                stats["fail"] += 1
                print(f"  [失败] {e}")
        elif obj_type and obj_type not in EXPORTABLE and obj_type:
            stats["skipped"] += 1

        if has_child and node_token:
            walk_wiki(token, space_id, node_token, current_path, file_ext, output_root, stats)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="飞书云文档批量导出")
    p.add_argument(
        "--token",
        required=True,
        help="user_access_token（个人文档推荐）或 tenant_access_token",
    )
    p.add_argument(
        "--source",
        choices=("drive", "wiki"),
        required=True,
        help="drive=云空间文件夹, wiki=知识库/我的文档库",
    )
    p.add_argument("--folder-token", help="云空间文件夹 token（source=drive 时必填）")
    p.add_argument(
        "--space-id",
        help="知识库 space_id；导出「我的文档库」用 my_library（source=wiki 时必填）",
    )
    p.add_argument("--output", required=True, type=Path, help="本地保存目录")
    p.add_argument(
        "--format",
        default="docx",
        choices=("docx", "pdf", "xlsx", "csv"),
        help="导出格式（默认 docx）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "ok": 0, "fail": 0, "skipped": 0}

    print(f"保存目录: {output_root}")
    print(f"导出格式: .{args.format}\n")

    if args.source == "drive":
        if not args.folder_token:
            # 空字符串表示根目录
            print("提示: 未指定 --folder-token，将导出云空间根目录")
        walk_drive(
            args.token,
            args.folder_token or None,
            Path("."),
            args.format,
            output_root,
            stats,
        )
    else:
        if not args.space_id:
            print("错误: wiki 模式需要 --space-id（我的文档库填 my_library）", file=sys.stderr)
            return 1
        walk_wiki(
            args.token,
            args.space_id,
            None,
            Path("."),
            args.format,
            output_root,
            stats,
        )

    print(
        f"\n完成: 共 {stats['total']} 个可导出文档, "
        f"成功 {stats['ok']}, 失败 {stats['fail']}, 跳过 {stats['skipped']}"
    )
    return 0 if stats["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
