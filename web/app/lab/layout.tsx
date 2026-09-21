import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Evaluation Lab | OpenPoke',
  description: 'Inspect aligned baseline and enhanced causal evidence from a paired evaluation run.',
};

export default function LabLayout({ children }: { children: React.ReactNode }) {
  return children;
}
