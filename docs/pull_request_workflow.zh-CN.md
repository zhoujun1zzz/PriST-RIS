# 本仓库提交 Pull Request

本页记录 2026-09-16 在 `zhoujun1zzz/PriST-RIS` 上实际验证过的提交方式。**SSH 用于 Git 拉取和推送；创建 PR 还需要已登录的 GitHub 网页或具有 API 权限的 GitHub CLI。** 能通过 SSH 推送分支，并不表示 GitHub API 连接器也能创建 PR。

## 已验证的路径：SSH 推送，网页创建 PR

在本地仓库中，从最新 `main` 建立工作分支；先检查当前改动，以免带入无关文件：

```bash
git status --short --branch
git switch main
git pull --ff-only origin main
git switch -c docs/your-change
# 编辑所需文件
git diff --check
git add -- path/to/changed-file
git commit -m "docs: describe the change"
git push -u origin docs/your-change
```

本仓库的 `origin` 使用 `git@github.com:zhoujun1zzz/PriST-RIS.git`。本次通过 SSH 成功查询 `main` 并推送 `docs/paper-v29-readme`。推送后，登录 GitHub 并打开 Git 输出的创建 PR 链接；也可以到仓库的 **Pull requests → New pull request**，选择 `base: main` 和 `compare: docs/your-change`。填写标题、改动原因和验证方式，检查 **Files changed** 后创建 PR。最后确认 PR 页面显示正确的目标分支和改动文件。不要把“分支已推送”误认为“PR 已创建”。

## 可选路径：GitHub CLI

`gh` 可以创建 PR，但它需要独立有效的 GitHub API 登录；Git 的 SSH 密钥不能代替这一步。先运行 `gh auth status`。若它显示令牌无效，使用 `gh auth login` 完成网页授权，再确认 `gh auth status` 正常。之后可运行：

```bash
gh pr create --base main --head docs/your-change --title "docs: describe the change" --body-file path/to/pr-body.md
gh pr view --web
```

PR 描述文件应写清具体问题、改动后的行为和验证结果。不要把访问令牌、私钥或服务器内部路径写进提交、PR 正文或命令历史。GitHub CLI 的[登录说明](https://cli.github.com/manual/gh_auth_login)和[创建 PR 说明](https://cli.github.com/manual/gh_pr_create)以官方文档为准。

## 本次遇到的不可行路径

- 已连接的 GitHub 读取接口可以读取仓库，但调用创建分支和创建 PR 时，GitHub 返回 `403 Resource not accessible by integration`。这是该连接器的写入权限问题，**不是 SSH 推送失败**；重复尝试相同接口不会解决。
- 当时本机 `gh auth status` 报告保存的令牌无效，因此不能直接依赖 `gh pr create`。重新完成 GitHub CLI 登录后才可使用它。
- 未登录的 GitHub 网页会把 PR 创建链接重定向到登录页。登录后再打开链接即可继续。

若在 Windows 上以另一账户运行 Git，遇到 `detected dubious ownership`，先确认确实是自己信任的本地仓库，再针对该次命令使用 `git -c safe.directory="<trusted-local-repo-path>" ...`；不要为了提交 PR 而盲目把所有目录加入全局信任列表。这个所有权提示与 GitHub 身份验证是两件事。
