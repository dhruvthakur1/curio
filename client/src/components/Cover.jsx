import { useState } from 'react';
import { coverGradient, shortTitle } from '../format.js';

/** Product image with a coloured fallback when Amazon's image is missing or blocked. */
export default function Cover({ item, className = '', style, label = false }) {
  const [failed, setFailed] = useState(false);
  const showImg = item.image && !failed;
  return (
    <span className={`cover ${className}`} style={style}>
      {showImg ? (
        <img src={item.image} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(true)} />
      ) : (
        <span className="cover-fallback" style={{ background: coverGradient(item.id) }}>
          {label ? shortTitle(item.title, 28) : ''}
        </span>
      )}
    </span>
  );
}
