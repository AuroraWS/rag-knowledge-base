#!/usr/bin/env bash
# ============================================================
# Claude Code Stop Hook — 交付验收
# 规则：如果本轮改了代码/配置/文档，但没有验证记录，则阻止结束。
# ============================================================
set -euo pipefail

TRANSCRIPT=$(cat)

# 收集变更文件
CHANGED=$(git diff --name-only HEAD 2>/dev/null || true)
STAGED=$(git diff --name-only --cached 2>/dev/null || true)
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null || true)

ALL="${CHANGED}
${STAGED}
${UNTRACKED}"

# 过滤出代码/配置/文档文件
RELEVANT=$(echo "$ALL" | grep -iE \
  '\.(py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|hpp|css|scss|html|vue|svelte|json|ya?ml|toml|ini|cfg|conf|md|rst|txt|mdx)$|^Dockerfile$|^Makefile$|\.env\.example$' \
  | sort -u | sed '/^$/d' || true)

if [ -z "$RELEVANT" ]; then
  exit 0
fi

# ============================================================
# 在 transcript 中搜索验证信号
# ============================================================
VERIFICATION_SIGNALS=(
  # 测试执行 & 结果
  'pytest'
  'npm test'
  'npm run test'
  'go test'
  'cargo test'
  'python -m pytest'
  'python -m unittest'
  'collected [0-9]+ items'
  '[0-9]+ passed'
  'All tests passed'
  'test(s)? passed'
  'tests? (run|executed|completed|finished)'
  '测试.*通过'
  '全部.*通过'
  # Lint
  'ruff check'
  'ruff\s'
  'flake8'
  'eslint'
  'pylint'
  'npx eslint'
  'All checks passed'
  'no (lint|style) (errors|issues|warnings)'
  '0 errors'
  'lint.*(通过|pass|clean|clear)'
  # 类型检查
  'mypy'
  'pyright'
  'npx tsc'
  'tsc --noEmit'
  'typescript.*(no error|clean|success)'
  'type.check.*(pass|success|no error|通过)'
  '0 (type )?errors? found'
  # 功能验证 / 手动测试
  '功能验证'
  '验证通过'
  '验证成功'
  'verified'
  'verification.*(done|complete|pass)'
  'manually tested'
  '手动测试'
  '确认.*正常'
  '功能.*正常'
  # TODO 检查
  'TODO.*(checked|done|complete|检查|完成)'
  'todo.*(all )?(done|clear|resolved|完成)'
  '所有.*(TODO|todo|待办).*(完成|处理|done|clear)'
  'no (remaining )?TODOs?'
  # Claude 明确声明验证完成
  '验证.*(完成|完毕|done|complete)'
  '所有.*验证.*(通过|pass)'
)

FOUND=false
for pattern in "${VERIFICATION_SIGNALS[@]}"; do
  if echo "$TRANSCRIPT" | grep -qiE "$pattern" 2>/dev/null; then
    FOUND=true
    break
  fi
done

if [ "$FOUND" = true ]; then
  exit 0
fi

# ============================================================
# 验证未通过 — 阻止结束
# ============================================================
cat << 'MSG'
╔══════════════════════════════════════════════════════════╗
║  🚫 交付验收未通过 — 请勿结束会话                          ║
╠══════════════════════════════════════════════════════════╣
║ 本轮修改了以下代码/配置/文档文件，但未检测到验证操作：       ║
╚══════════════════════════════════════════════════════════╝
MSG

echo "$RELEVANT" | while read -r f; do
  echo "  - $f"
done

cat << 'MSG'

请继续工作，完成以下至少一项验证后再结束：
  1. 运行测试并报告结果
  2. 运行 Lint / 类型检查并报告结果
  3. 进行功能验证并输出结果
  4. 检查 TODO 并确认全部处理完毕

验证完成后，再次尝试结束会话。

MSG

exit 1
