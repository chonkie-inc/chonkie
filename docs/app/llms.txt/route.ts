import { llms } from 'fumadocs-core/source/llms';
import { source } from '@/lib/source';

export const dynamic = 'force-static';

const lm = llms(source);

export function GET() {
  return new Response(lm.index(), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
